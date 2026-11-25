//! Stores through title memory by address (`shader.rs` `SLOT_MEMORY`), run
//! on the card: the dword lands in the copy the directory names for the
//! page, and the store marks that copy for the host (`MEMORY_MARKS`), for
//! the lanes that run and no others.

use super::host::{Dispatch, Host, Root, View};
use super::{
    HostShader, MEMORY_FAULTS, MEMORY_MARKS, SLOT_FAULTS, SLOT_MEMORY, USER_MEMORY_BASE, lower,
    validator,
};
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo};
use crate::{io, ir, resource, structure};

/// Threads of the one group.
const THREADS: u32 = 32;
/// Threads that store: the rest are masked out before the store.
const STORING: u32 = 24;
/// The title address the first thread stores to: page 2, byte 0x40.
const ADDRESS: u32 = 0x0002_0040;
/// What thread `t` stores is this plus `t`.
const VALUE: u32 = 0x1000;

/// Each thread below [`STORING`] stores [`VALUE`] + its number at
/// [`ADDRESS`] + 4 * its number, an address it builds in a register pair.
fn program() -> Vec<u32> {
    vec![
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x4A02_02FF,
        ADDRESS,     // v_add_nc_u32 v1, ADDRESS, v1
        0x7E04_0280, // v_mov_b32 v2, 0
        0x4A06_00FF,
        VALUE,                          // v_add_nc_u32 v3, VALUE, v0
        0x7DA8_0000 | (0x80 + STORING), // v_cmpx_gt_u32 exec_lo, STORING, v0
        0xDC70_8000,
        0x007D_0301, // global_store_dword v3, v[1:2] (no scalar base)
        0xBF81_0000, // s_endpgm
    ]
}

fn native(words: &[u32]) -> NativeProgram {
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn build() -> HostShader {
    let native = native(&program());
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("tracked");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("the program lowers");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// Runs the program with page 2 in copy `copy` of the directory (0: no
/// copy) and returns the three writable views of the heap: the view at the
/// base slot (where accesses without a copy go), copy 1, and the fault
/// notes with the marks after them.
fn run(host: &Host, shader: &HostShader, copy: u32) -> Vec<Vec<u32>> {
    // The view heap slot the copies' views start at is the one user-data
    // dword the host computes; the rest is the title's (unused here).
    let mut user = vec![0u32; (shader.user_data_dwords() as usize).max(1)];
    for u in &shader.user_data {
        match u.root {
            RootType::Other(USER_MEMORY_BASE) => user[u.index as usize] = 0,
            RootType::Constants => {}
            other => panic!("unexpected user data {other:?}"),
        }
    }
    // Root views follow the user data by slot: the faults (a UAV) at
    // [`SLOT_FAULTS`], then the directory (an SRV) at [`SLOT_MEMORY`].
    const _: () = assert!(SLOT_FAULTS < SLOT_MEMORY);
    // The directory: two dwords a page (the copy's number, the page's byte
    // offset in the copy), up to page 2.
    let directory = vec![0, 0, 0, 0, copy, 0];
    let (root, pso) = host.compute_pipeline(shader).expect("compute_pipeline");
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![
                (0, Root::Constants(user)),
                (1, Root::Written(2)),
                (
                    2,
                    Root::View {
                        srv: true,
                        data: directory,
                    },
                ),
            ],
            views: &[
                View::RawUav(64),
                View::RawUav(64),
                View::RawUav(MEMORY_FAULTS + MEMORY_MARKS),
            ],
            samplers: &[],
            groups: [1, 1, 1],
        },
    )
    .expect("dispatch")
}

#[test]
fn a_store_by_address_lands_in_its_copy_and_marks_it_on_the_card() {
    let shader = build();
    assert!(shader.root_srvs.contains(&SLOT_MEMORY));
    assert!(shader.root_uavs.contains(&SLOT_FAULTS));
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let first = (ADDRESS & 0xFFFF) as usize / 4;
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        ran += 1;

        // Page 2 is in copy 1.
        let out = run(&host, &shader, 1);
        let (none, copy, notes) = (&out[0], &out[1], &out[2]);
        for t in 0..THREADS as usize {
            let want = if t < STORING as usize {
                VALUE + t as u32
            } else {
                0
            };
            assert_eq!(
                copy[first + t],
                want,
                "{label}: thread {t}'s dword in the copy"
            );
        }
        assert!(
            none.iter().all(|&d| d == 0),
            "{label}: nothing at the base slot"
        );
        let (faults, marks) = notes.split_at(MEMORY_FAULTS as usize);
        // Dword 0 is the sink of the lanes that do not run.
        assert!(
            faults[1..].iter().all(|&d| d == 0),
            "{label}: no fault note"
        );
        assert_eq!(marks[1], 1, "{label}: copy 1 is marked");
        assert!(
            marks.iter().enumerate().all(|(n, &d)| n == 1 || d == 0),
            "{label}: no other mark"
        );

        // Page 2 has no copy: the stores are dropped at the base slot, the
        // page is noted, and no copy is marked (mark 0 is no copy's).
        let out = run(&host, &shader, 0);
        let (copy, notes) = (&out[1], &out[2]);
        assert!(copy.iter().all(|&d| d == 0), "{label}: copy 1 untouched");
        let (faults, marks) = notes.split_at(MEMORY_FAULTS as usize);
        let page = ADDRESS >> 16;
        let place = (page % (MEMORY_FAULTS - 1) + 1) as usize;
        assert_eq!(faults[place], page, "{label}: the page is noted");
        assert!(
            marks[1..].iter().all(|&d| d == 0),
            "{label}: no copy marked"
        );
    }
    if ran == 0 {
        eprintln!("no device ran the store-by-address test");
    }
}

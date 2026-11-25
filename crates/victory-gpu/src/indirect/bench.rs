//! 3,200 one-record draws clamped individually or in windows of 32. Compares
//! per-record charging, PR #780's scratch handling, grouped charging, and
//! grouped charging with a device-local budget. Run the ignored test with
//! `cargo test -p victory-gpu --release --lib indirect::bench -- --ignored --nocapture`.
//! Add `--features diagnostics` and set `VICTORY_D3D12_WARP=1` to use WARP.
//! The test draws its own triangles and needs no title files.

use super::*;
use windows::Win32::Foundation::{CloseHandle, RECT};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR};

/// Draws a frame (the title's gameplay frames hold about 3,200 indirect
/// executions, #107), records a window holds, and how far apart the
/// records are (the title keeps them 32 bytes apart).
const DRAWS: u32 = 3200;
const WINDOW: u32 = 32;
const STEP: u32 = 32;
const RUNS: usize = 9;

const SHADERS: &str = r#"
#define RS "RootFlags(0)"
[RootSignature(RS)]
float4 vs(uint v : SV_VertexID, uint i : SV_InstanceID) : SV_Position {
float2 p = float2((v & 1) ? 0.05 : 0.0, (v & 2) ? 0.05 : 0.0) - 0.5;
return float4(p, 0.5, 1);
}
float4 ps() : SV_Target { return float4(1, 0.5, 0.25, 1); }
"#;

fn compile(entry: &str, target: &str) -> Vec<u8> {
    let entry = std::ffi::CString::new(entry).unwrap();
    let target = std::ffi::CString::new(target).unwrap();
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names outlive the call.
    unsafe {
        D3DCompile(
            SHADERS.as_ptr().cast(),
            SHADERS.len(),
            s!("bench"),
            None,
            None,
            PCSTR(entry.as_ptr().cast()),
            PCSTR(target.as_ptr().cast()),
            0,
            0,
            &mut code,
            None,
        )
        .unwrap();
        let code = code.unwrap();
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
            .to_vec()
    }
}

#[test]
#[ignore = "benchmark; see the module notes"]
fn bench_indirect_draw_budget() {
    let Some(host) = super::host() else {
        eprintln!("no host GPU");
        return;
    };
    let device = &host.device;
    let clamps = [
        make_for_with(device, cfg!(feature = "diagnostics"), false).expect("per-record clamp"),
        make_for_with(device, cfg!(feature = "diagnostics"), true).expect("grouped clamp"),
    ];
    // SAFETY: plain D3D12 object creation with valid descriptions; the
    // objects outlive every list recorded and run below.
    unsafe {
        let adapter = {
            let luid = device.GetAdapterLuid();
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| {
                    f.EnumAdapterByLuid::<windows::Win32::Graphics::Dxgi::IDXGIAdapter1>(luid)
                })
                .and_then(|a| a.GetDesc1())
                .map(|d| {
                    String::from_utf16_lossy(&d.Description)
                        .trim_end_matches('\0')
                        .to_owned()
                })
                .unwrap_or_else(|_| "an unnamed adapter".to_owned())
        };
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .unwrap();
        list.Close().unwrap();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let event = CreateEventW(None, false, false, None).unwrap();
        let frequency = queue.GetTimestampFrequency().unwrap() as f64;

        // The pipeline: tiny triangles into a small target.
        let (vs, ps) = (compile("vs", "vs_5_1"), compile("ps", "ps_5_1"));
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs).unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.as_ptr().cast(),
                BytecodeLength: vs.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.as_ptr().cast(),
                BytecodeLength: ps.len(),
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
            D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
        let target_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut target: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &target_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut target,
            )
            .unwrap();
        let target = target.unwrap();
        let rtvs: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap();
        let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
        device.CreateRenderTargetView(&target, None, rtv);

        // The title's records: three vertices, one instance, 32 bytes apart.
        let (src, src_ptr) = host.shared_buffer(u64::from(DRAWS * STEP)).unwrap();
        std::ptr::write_bytes(src_ptr, 0, (DRAWS * STEP) as usize);
        for i in 0..DRAWS as usize {
            let words = [3u32, 1, 0, 0];
            std::ptr::copy_nonoverlapping(
                words.as_ptr().cast::<u8>(),
                src_ptr.add(i * STEP as usize),
                16,
            );
        }
        let (budget, budget_ptr) = host.shared_buffer(256).unwrap();
        let budget_gpu = budget.GetGPUVirtualAddress();
        let card_budget = budget_writer(device, &list).map(|writer| {
            let buffer = host.uav_buffer(256).unwrap();
            let address = buffer.GetGPUVirtualAddress();
            (buffer, writer, address)
        });
        let scratch = host
            .uav_buffer(u64::from(WINDOW * STEP).max(64 << 10))
            .unwrap();
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: 16,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
            sig.unwrap()
        };
        let stamps: ID3D12QueryHeap = {
            let mut heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .unwrap();
            heap.unwrap()
        };
        let mut back: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 16,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )
            .unwrap();
        let back = back.unwrap();
        let mut submitted = 0u64;

        // One frame's lists: GPU ms (timestamps around the list), CPU ms
        // recording it, ms from submission to the end of its run.
        let mut frame = |keep: bool, clamp: &Clamp, window: u32, card: bool| -> [f64; 3] {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
            let recording = std::time::Instant::now();
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            let budget_address = if card {
                let (buffer, writer, address) = card_budget.as_ref().unwrap();
                zero_card_budget(writer, buffer);
                *address
            } else {
                budget_gpu
            };
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 256.0,
                Height: 256.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 256,
                bottom: 256,
            }]);
            list.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            list.SetGraphicsRootSignature(&root);
            for first in (0..DRAWS).step_by(window as usize) {
                // PR #780 leaves scratch readable until the next clamp.
                if keep && first > 0 {
                    done_on(&list, &scratch);
                }
                let records = window.min(DRAWS - first);
                assert!(record_clamp_as(
                    clamp,
                    &list,
                    &scratch,
                    budget_address,
                    (&src, u64::from(first * STEP)),
                    None,
                    (0, STEP, records),
                    window > 1,
                ));
                list.SetPipelineState(&pso);
                for k in 0..records {
                    list.ExecuteIndirect(&signature, 1, &scratch, u64::from(k * STEP), None, 0);
                    if !keep {
                        done_on(&list, &scratch);
                    }
                }
            }
            if card {
                let (buffer, _, _) = card_budget.as_ref().unwrap();
                list.ResourceBarrier(&[transition(
                    buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
            list.Close().unwrap();
            let recorded = recording.elapsed().as_secs_f64() * 1e3;
            assert_eq!(host.log_messages(), None, "invalid benchmark recording");
            let running = std::time::Instant::now();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            submitted += 1;
            queue.Signal(&fence, submitted).unwrap();
            fence.SetEventOnCompletion(submitted, event).unwrap();
            WaitForSingleObject(event, INFINITE);
            let ran = running.elapsed().as_secs_f64() * 1e3;
            device.GetDeviceRemovedReason().unwrap();
            assert_eq!(host.log_messages(), None, "invalid benchmark submission");
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data)).unwrap();
            let t = std::slice::from_raw_parts(data.cast::<u64>(), 2);
            let gpu = t[1].saturating_sub(t[0]) as f64 / frequency * 1e3;
            back.Unmap(0, None);
            [gpu, recorded, ran]
        };

        let (old, new) = (&clamps[0], &clamps[1]);
        let mut cases = vec![
            ("single, per record", false, old, 1, false),
            ("single, keep scratch", true, old, 1, false),
            ("single, grouped", true, new, 1, false),
            ("window, per record", false, old, WINDOW, false),
            ("window, keep scratch", true, old, WINDOW, false),
            ("window, grouped", true, new, WINDOW, false),
        ];
        if card_budget.is_some() {
            cases.extend([
                ("single, grouped on card", true, new, 1, true),
                ("window, grouped on card", true, new, WINDOW, true),
            ]);
        }
        // Warm up, then the cases in turn, so drift falls on all alike.
        for &(_, keep, clamp, window, card) in &cases {
            frame(keep, clamp, window, card);
        }
        let mut times: Vec<Vec<[f64; 3]>> = vec![Vec::new(); cases.len()];
        for _ in 0..RUNS {
            for (i, &(_, keep, clamp, window, card)) in cases.iter().enumerate() {
                times[i].push(frame(keep, clamp, window, card));
            }
        }
        let median = |v: &[[f64; 3]], k: usize| {
            let mut x: Vec<f64> = v.iter().map(|t| t[k]).collect();
            x.sort_by(f64::total_cmp);
            x[x.len() / 2]
        };
        eprintln!(
            "indirect draw budgets on {adapter}: {DRAWS} one-record draws a frame, \
             medians of {RUNS} frames"
        );
        for (i, (name, ..)) in cases.iter().enumerate() {
            let gpu = median(&times[i], 0);
            eprintln!(
                "  {name:<22} GPU {gpu:7.2} ms ({:5.2} us a draw), recording {:6.2} ms, \
                 submit to done {:7.2} ms",
                gpu * 1e3 / f64::from(DRAWS),
                median(&times[i], 1),
                median(&times[i], 2),
            );
        }
        CloseHandle(event).unwrap();
    }
}

# Adaptive inline movie texture copies

Related to #39, #269, #80, #159, #343 and #438. Credit to
arminkhaleghirod, NuraadGit, PAYS0, adritoyalser, brunofgmag, and
xunbreakablex (two-change probe suggestion in #39). This is a
standalone experiment on a803c87c, not a replacement for alias-write
notification (#438) or staging lifetime synchronization (#272).

Record-time clean checks cannot predict submission-time refresh.
Keep the original inline copy point for recently changing textures.
A full snapshot's first change enables a follow-up hash/copy;
a second change at a distinct check frame within 24 frames keeps
the texture hot. Static textures go back to the normal policy.
Fully mapped direct, copied and successful unlocked reads all
participate. Partial mapping retains its existing polling policy.
No texture resolution, draw count or filtering policy changes.

Set VICTORY_TEXTURE_MOTION=0 for baseline. Compare the same startup
logo segment and world checkpoint, warm caches, no capture for timing.
Report repeated/backward images separately from frame intervals.
Start with relay off, then test relay independently. A first missed
change can still appear late; in-flight staging races remain outside
this experiment. Extra copies/hashes of changing textures may cost
CPU/GPU time. Native Rust/GPU tests pending; no measured improvement.
AI-assisted (ChatGPT).

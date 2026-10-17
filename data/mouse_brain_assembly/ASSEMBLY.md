# Mouse-brain mouse — assembly instructions

A computer mouse whose case is a 3D-printed mouse brain, built around the
Bambu Lab Wireless Mouse Components Kit-002 (MH002) electronics.

**Print time:** ~10 h body + ~1 h plate (PETG, 0.16 mm layers, 25% infill)
**Skill level:** intermediate. Requires a #0 Phillips driver for M2 screws.

---

## Parts & tools

### Printed parts
| file | print | notes |
|---|---|---|
| `one_piece.stl` | PETG, 0.16 mm, 25% infill, tree supports | Body + L/R cortex + 2 living hinges. Print flat-cut side DOWN on build plate. **Hinge crucial**: use 100% infill in the hinge zone, 245 °C nozzle, 30 mm/s through hinge, max 30% fan. |
| `bottom_plate.stl` | PETG, 0.16 mm, 25% infill | Flat — no supports needed |

### Kit electronics (Bambu Lab MH002)
1. PCB (with on-board L/R click microswitches, optical sensor, wheel encoder, toggle/mute switch)
2. Scroll wheel + wheel bracket
3. AA battery contact rail (with spring at one end)
4. Battery cradle (the moulded plastic part with battery contacts)
5. USB receiver dongle
6. **6× M2×8 self-tapping screws (1 spare = 5 used: 4 for the bottom plate, kit's own PCB attachment is now handled by plate compression)**
7. 4× PTFE mouse skates

### User-supplied
- 1× AA battery
- #0 Phillips screwdriver

---

## Pre-assembly checks (do these BEFORE printing again if any fail)

1. **Hinge flex test**: after printing `one_piece.stl`, gently lift each cortex hemisphere 1–2 mm at the front. The hinge tab at the back should flex without cracking. If it cracks → reprint with 100% infill in hinge + slower print speed.
2. **Plate fits in rebate**: drop the plate into the body's rebate. It should slide in with a slight clearance fit (0.6 mm per side). If too tight, sand the edges 0.2 mm.
3. **Wheel slot fits the wheel**: dry-fit the wheel into the wheel slot. It should rotate freely with 1.5 mm clearance. If binding → file the slot edges.

---

## Step 1: Pre-set the toggle/mute switch

**Critical pre-assembly step.** The toggle switch (mute) at PCB position (11.5, 5.4, 8.9) lives **inside** the body cavity after assembly and **cannot be reached without removing the bottom plate**.

➜ **Set the toggle to ON now**, before any further assembly.

> *Design note: in V1 the toggle is intentionally not exposed (no side slot in the brain — would disrupt the anatomy). To switch OFF later, you must remove the bottom plate.*

## Step 2: Connect the wheel bracket to the PCB

The kit's wheel bracket and encoder plate mount the scroll wheel to the PCB. Follow Bambu's kit instructions: snap the wheel into the bracket, clip the bracket onto the PCB. Verify the wheel rotates freely on its axle.

## Step 2.5: Snap off the 8 scaffold tabs

Before installing electronics, you must release the cortex buttons.

The cortex hemispheres were printed with **4 breakaway scaffold tabs each** (0.6 mm thick, located at y≈5 around each hemisphere's perimeter) so they couldn't sag in the 2 mm pocket gap during printing. With the body flat-side UP, look down through the chassis opening:

1. You'll see 8 thin tab connections between the cortex outer wall and the body's pocket wall.
2. Insert a small screwdriver / fingernail and push each tab sideways until it snaps. Each tab takes ~5 N of force.
3. After all 8 are broken, the cortex hemispheres should pivot freely on their living hinges.

✓ Sanity check: press the front of each cortex hemisphere — it should depress 0.25-0.5 mm and spring back. Free movement = scaffolds fully removed.

## Step 3: Install the PCB into the body cavity

Hold the body with the flat-cut side facing UP (chassis opening visible).

1. Lower the PCB **sensor-side down** into the chassis opening. The 22 mm scroll wheel must pass through the wheel slot (between the two cortex hemispheres at the front).
2. Align the optical sensor (under the PCB) over the sensor-window position. The lens points DOWN.
3. The PCB will sit on the cavity floor; the battery cradle (large plastic part on PCB +z side) rests against the body's right wall.

✓ Sanity check: looking down through the chassis opening, you should see the PCB's underside (with the sensor lens visible). Looking at the dorsal (top) side of the brain, the wheel should poke up between the two cortex hemispheres.

## Step 4: Install the AA battery

1. Insert one end of the AA against the **spring contact** in the battery cradle (push to compress the spring).
2. Drop the other end into the fixed contact.
3. The battery should sit horizontally in the cradle, along the AP axis at the back-left of the chassis. Total length ~49 mm.

✓ Sanity check: battery seated firmly. No movement when you wiggle the body.

## Step 5: Install the bottom plate

1. Lower the plate into the body's rebate (the 2.5 mm-deep recess in the base). It should drop in flush.
2. **Note** the four PCB-retention bumps on the plate's top surface. As the plate seats, these contact the PCB underside and clamp the PCB upward against the cavity (0.4 mm compression).
3. Drive the 4× M2×8 screws through the countersunk holes in the plate corners. Hand-tight + ¼ turn. Don't over-torque (PETG threads strip easily).

✓ Sanity check: plate sits flush with the body bottom. Screw heads countersunk flush (not raised). Mouse sits flat on a table — wobble-free.

## Step 6: Apply skates

Stick 4× PTFE skates to the recessed pads on the plate's underside (one at each corner inset 10 mm from the plate edge). Press firmly.

## Step 7: Test

1. Pop the USB dongle out of its storage cup (under the battery, accessed by removing the plate) and plug it into your computer.
2. With the mute toggle ON (set in Step 1), pressing either cortex hemisphere should generate a click event.
3. Roll the scroll wheel between the hemispheres.
4. Move the mouse — the cursor should track via the optical sensor.

---

## Maintenance

### Battery replacement (the common case)
1. Unscrew 4× M2 screws.
2. Lift the bottom plate. PCB stays in place (light friction fit + the cavity walls).
3. Pop the AA out by pushing it against the spring contact.
4. Insert fresh AA.
5. Replace plate, screw down.

### Toggle the mute (rare)
Same as battery replacement, then flip the toggle on the PCB.

### Replace PCB
Lift the PCB straight up out of the chassis opening (after removing battery, then plate).

---

## Known limitations (V1)

| limitation | reason | mitigation |
|---|---|---|
| Toggle switch hidden | Adding a side slot would disrupt the brain anatomy | Pre-set ON; remove plate to switch |
| USB dongle storage under plate | Same reason — no external dongle slot | Remove plate to access (1× per use) |
| Cross-layer hinge fatigue (PETG) | Hinge flexes perpendicular to print layers | Use settings above; ~1000 click life expected; reprint cortex if needed |
| Wide footprint (148 × 116 mm) | Brain LR dimension dictates width | This is the natural brain shape; lighter palm grip works best |

---

## Validated mechanics (all PASS, see `validate_mechanics.py`)

| mechanic | result |
|---|---|
| Cortex print clearance — 2 mm gap + 8 breakaway scaffolds for PETG print | ✓ |
| Click — 0.25 mm front travel engages switch at 0.05° pivot | ✓ |
| Screw access — 4 bosses ≥2.5 mm clear of opening, screwdriver path clear | ✓ |
| Plate removal — 0.6 mm side clearance, vertical rebate walls, no undercut | ✓ |
| Battery removal — straight-down extraction, 0.6% peak contact, fully clear at 12 mm | ✓ |
| Sensor access — lens window aligned, clear path to desk | ✓ |
| PCB retention — 4 plate bumps clamp PCB with 0.4 mm compression | ✓ |

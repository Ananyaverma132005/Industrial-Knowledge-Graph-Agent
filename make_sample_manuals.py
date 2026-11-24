"""Generates sample maintenance manuals (PDF) for the demo.

File-name prefix tells ingest_pdf.py which machines a manual applies to:
  M1_...pdf -> M1,   M2_M6_...pdf -> M2 and M6,   ALL_...pdf -> every machine.
Replace these with real plant manuals for production use.

    python make_sample_manuals.py
"""
from fpdf import FPDF

import config

MANUALS = {
    "M1_cnc_milling_manual.pdf": ("CNC Milling Machine (M1) - Maintenance Manual", [
        ("SOP-CNC-01 Spindle Overheating Response",
         "Trigger: spindle temperature above 90 deg C or thermal alarm on the controller.\n"
         "1. Feed-hold the program and let the spindle run unloaded at 20 percent speed for 5 minutes to cool evenly; do not stop it abruptly.\n"
         "2. Check coolant flow at the nozzle; flow below 8 L/min points to the coolant pump (see SOP-CNC-03).\n"
         "3. Inspect the spindle chiller set point (target 22 deg C) and clean the chiller air filter.\n"
         "4. Verify spindle bearing lubrication: the oil-air unit must show one pulse every 8 minutes.\n"
         "5. Run the spindle warm-up cycle (O9001) before resuming production.\n"
         "6. If the temperature exceeds 90 deg C again within 2 hours, raise a ticket and inspect bearing preload."),
        ("SOP-CNC-02 Spindle Vibration Diagnosis",
         "Trigger: vibration above 7.1 mm/s RMS or poor surface finish.\n"
         "1. Check tool holder runout with a dial indicator; it must be below 5 microns.\n"
         "2. Clean the spindle taper and the tool holder flange with a lint-free cloth.\n"
         "3. Balance the tool assembly if spindle speed is above 8000 rpm.\n"
         "4. Run the vibration spectrum test; a peak at bearing frequency means the bearings need replacement by a CNC technician."),
        ("SOP-CNC-03 Coolant Pump Priming Procedure",
         "Trigger: low coolant flow alarm or pump not priming at shift start.\n"
         "1. Apply lockout/tagout to the coolant system (see SOP-SAF-01).\n"
         "2. Check the coolant tank level; top up with 6 percent emulsion if below the MIN mark.\n"
         "3. Open the bleed screw on the pump housing until coolant flows without air bubbles, then close it.\n"
         "4. Clean the suction strainer of chips and sludge.\n"
         "5. Remove the lockout, run the pump for 2 minutes and confirm flow above 8 L/min."),
        ("SOP-CNC-04 Servo Motor Fault Reset",
         "Trigger: servo alarm or axis overload on the CNC controller.\n"
         "1. Record the alarm code and the axis.\n"
         "2. Check the axis for mechanical obstruction and the ball screw for chips.\n"
         "3. Inspect the servo motor cable connector for looseness or coolant ingress.\n"
         "4. Power-cycle the drive and reset the alarm; if it repeats within one shift, replace the encoder cable."),
    ]),
    "M3_hydraulic_press_manual.pdf": ("Hydraulic Press (M3) - Maintenance Manual", [
        ("SOP-HYD-01 Hydraulic Seal Replacement",
         "Trigger: oil leakage at the ram, or pressure drop below 170 bar under load.\n"
         "1. Lower the ram onto safety blocks and apply lockout/tagout (SOP-SAF-01).\n"
         "2. Release stored hydraulic pressure through the accumulator dump valve; confirm 0 bar on the gauge.\n"
         "3. Remove the gland nut and extract the worn seal set with a non-metallic pick.\n"
         "4. Inspect the cylinder rod for scoring; polish minor marks, replace the rod if scoring is deeper than 0.1 mm.\n"
         "5. Install the new seal kit lubricated with clean hydraulic oil ISO VG 46.\n"
         "6. Torque the gland nut to 180 Nm, bleed air from the cylinder and test at 210 bar for 10 minutes.\n"
         "7. Repeated seal failure within 90 days indicates contaminated oil: send an oil sample for analysis."),
        ("SOP-HYD-02 Hydraulic Pump Cavitation",
         "Trigger: whining noise from the pump or foaming oil.\n"
         "1. Check the reservoir oil level and top up to the sight-glass mark.\n"
         "2. Inspect the suction line and clamps for air leaks.\n"
         "3. Replace the suction filter if the differential pressure indicator is red.\n"
         "4. Verify oil temperature is between 40 and 55 deg C; cold oil causes cavitation at start-up."),
        ("SOP-HYD-03 Pressure Valve Calibration",
         "1. Install a calibrated test gauge on port P1.\n"
         "2. Adjust the relief valve until the system relieves at 210 bar plus or minus 2 bar.\n"
         "3. Lock the adjustment screw and record the setting in the maintenance log."),
    ]),
    "M2_M6_robotics_manual.pdf": ("Robotic Cells (M2, M6) - Maintenance Manual", [
        ("SOP-ROB-01 Servo Motor Overcurrent Fault",
         "Trigger: overcurrent or overload alarm on a robot axis.\n"
         "1. Put the robot in T1 (manual reduced speed) mode and jog the faulted axis slowly.\n"
         "2. Check for collisions, cable snags or a payload heavier than the configured value.\n"
         "3. Measure motor winding insulation with a megger; it must be above 100 megaohm.\n"
         "4. Inspect the axis gearbox for backlash.\n"
         "5. If the fault repeats, replace the servo motor and run the payload identification routine."),
        ("SOP-ROB-02 Encoder Error and Re-homing",
         "Trigger: encoder error or robot lost home position.\n"
         "1. Check the encoder battery voltage on the controller; replace if below 3.0 V.\n"
         "2. Jog each axis to its witness mark.\n"
         "3. Run the mastering (re-homing) routine from the teach pendant and verify with the TCP check point."),
        ("SOP-ROB-03 Gripper Vacuum Loss",
         "1. Check the vacuum generator supply air pressure, required 6 bar.\n"
         "2. Inspect suction cups for tears and replace worn cups.\n"
         "3. Clean the vacuum filter and test that vacuum reaches minus 0.6 bar within 0.3 seconds."),
        ("SOP-ROB-04 Welding Torch Nozzle Cleaning",
         "1. Move the torch to the cleaning station.\n"
         "2. Run the reamer cycle and spray anti-spatter fluid.\n"
         "3. Replace the contact tip if the bore is worn oval, typically every 8 hours of arc time."),
        ("SOP-ROB-05 Wire Feeder Slip Correction",
         "1. Check drive roll groove size matches the wire diameter.\n"
         "2. Set drive roll tension so the wire stops when pinched at the gun.\n"
         "3. Blow out the liner with dry air and replace it if kinked."),
    ]),
    "M4_conveyor_manual.pdf": ("Belt Conveyor (M4) - Maintenance Manual", [
        ("SOP-CNV-01 Drive Belt Alignment and Tensioning",
         "Trigger: belt fraying, drifting or slipping.\n"
         "1. Apply lockout/tagout to the conveyor drive (SOP-SAF-01).\n"
         "2. Check pulley alignment with a straight edge; adjust the take-up screws equally on both sides.\n"
         "3. Set belt tension so mid-span deflection is 1 percent of the span length.\n"
         "4. Run empty for 10 minutes and confirm the belt tracks centrally."),
        ("SOP-CNV-02 Drive Belt Replacement",
         "1. Lockout the drive and release tension at the take-up.\n"
         "2. Remove the old belt and inspect pulleys and rollers for wear or seized bearings.\n"
         "3. Fit the new belt, re-tension per SOP-CNV-01 and record the change date."),
    ]),
    "M5_cnc_lathe_manual.pdf": ("CNC Lathe (M5) - Maintenance Manual", [
        ("SOP-LTH-01 Chuck Jaw Clamping Check",
         "1. Clean the jaws and the chuck body serrations.\n"
         "2. Grease the chuck through the nipples with chuck paste, 2 strokes each.\n"
         "3. Measure clamping force with a gauge; it must be within 10 percent of the rated value.\n"
         "4. Bore soft jaws again if runout exceeds 0.02 mm."),
        ("SOP-LTH-02 Lathe Spindle Thermal Management",
         "1. Run the warm-up program for 15 minutes at the start of each shift.\n"
         "2. Limit continuous running at maximum speed to 4 hours, then allow a 10 minute idle.\n"
         "3. Check the spindle oil cooler and coolant concentration if temperature exceeds 85 deg C."),
    ]),
    "ALL_safety_manual.pdf": ("Assembly Line - General Safety Procedures", [
        ("SOP-SAF-01 Lockout/Tagout (LOTO)",
         "1. Inform operators and stop the machine with the normal stop button.\n"
         "2. Isolate every energy source: electrical, pneumatic, hydraulic.\n"
         "3. Apply your personal lock and tag at each isolation point.\n"
         "4. Release stored energy and verify zero energy by attempting a start.\n"
         "5. Only the person who applied a lock may remove it."),
        ("SOP-SAF-02 Personal Protective Equipment",
         "Safety shoes, safety glasses and hearing protection are mandatory on the line. "
         "Use cut-resistant gloves when handling tools or chips, and a welding helmet near the welding cell."),
        ("SOP-SAF-03 Emergency Stop and Restart",
         "1. Press the nearest emergency stop on any hazard.\n"
         "2. Report the event to the shift supervisor.\n"
         "3. Restart only after the cause is cleared and the supervisor resets the safety circuit."),
    ]),
}


def build_pdf(path, title, sections) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    nl = {"new_x": "LMARGIN", "new_y": "NEXT"}  # return to the left margin after each cell
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 9, title, **nl)
    pdf.ln(4)
    for heading, body in sections:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 7, heading, **nl)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5.5, body, **nl)
        pdf.ln(3)
    pdf.output(str(path))


def main() -> None:
    config.MANUALS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, (title, sections) in MANUALS.items():
        build_pdf(config.MANUALS_DIR / fname, title, sections)
        print(f"[manuals] wrote {fname} ({len(sections)} SOPs)")


if __name__ == "__main__":
    main()

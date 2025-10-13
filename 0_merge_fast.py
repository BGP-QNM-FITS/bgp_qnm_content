import json
from pathlib import Path

MAIN_DIR = Path("mode_content_files")
COMPILED_DIR = Path("mode_content_compiled")
SIM_ID = "0010"

main_file = MAIN_DIR / f"mode_content_data_{SIM_ID}_D.json"
merge_file = COMPILED_DIR / f"mode_content_data_{SIM_ID}_2.json"
output_file = MAIN_DIR / f"mode_content_data_{SIM_ID}_merged.json"

def merge_main_with_compiled(main_data, comp_data):

    main_times = list(main_data.get("times", []))
    main_modes = list(main_data.get("modes", []))

    comp_times = list(comp_data.get("times", []))
    comp_modes = list(comp_data.get("modes", []))

    if len(main_times) != len(main_modes):
        raise RuntimeError("main file: times and modes length mismatch")
    if len(comp_times) != len(comp_modes):
        raise RuntimeError("compiled file: times and modes length mismatch")

    main_sph = list(main_data.get("spherical_modes", []))
    main_sph_set = set(tuple(sm) for sm in main_sph)
    comp_sph_set = set(tuple(sm) for sm in comp_data.get("spherical_modes", []))

    # Add the new spherical modes into the main spherical mode set 

    for sm in comp_sph_set:
        if sm not in main_sph_set:
            main_sph.append(list(sm))
            main_sph_set.add(sm)

    main_data["spherical_modes"] = main_sph

    # For each compiled (time, qnms) override/insert into main

    for t_comp, qnms_comp in zip(comp_times, comp_modes):

        comp_q_tuples = [tuple(q) for q in qnms_comp]

        if t_comp in main_times:
            i_main = main_times.index(t_comp)
            existing_q = main_modes[i_main]
            filtered = []
            for q in existing_q:
                if len(q) == 4:
                    if tuple(q[0:2]) in comp_sph_set:
                        continue 
                elif len(q) == 2:
                    None # keep the constant offsets - again this is a fix not a general behaviour! 
                elif len(q) != 4:
                    continue 
                    # Nonlinear modes removed by default!! Current lists definitely need overriding but 
                    # this is not general! 
                filtered.append(tuple(q))
            # add compiled qnms (avoid duplicates)
            for q in comp_q_tuples:
                if q not in filtered:
                    filtered.append(q)
            # store as lists
            main_modes[i_main] = [list(q) for q in filtered]
            print(f"Overrode time {t_comp}: removed {len(existing_q)-len(filtered)+len([q for q in comp_q_tuples if q not in existing_q])} / updated entries")
        else:
            # insert new time preserving sorted order of times
            # find insert position to keep main_times sorted
            insert_pos = 0
            while insert_pos < len(main_times) and main_times[insert_pos] < t_comp:
                insert_pos += 1
            main_times.insert(insert_pos, t_comp)
            # filtered: start from existing list empty then add compiled qnms (dedupe trivially)
            main_modes.insert(insert_pos, [list(q) for q in comp_q_tuples])
            print(f"Inserted new time {t_comp} with {len(comp_q_tuples)} qnms at pos {insert_pos}")

    # assign back
    main_data["times"] = main_times
    main_data["modes"] = main_modes
    return main_data


def main():
    with open(main_file, "r") as f:
        main_data = json.load(f)
    with open(merge_file, "r") as f:
        comp_data = json.load(f)

    merged = merge_main_with_compiled(main_data, comp_data)

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Wrote merged file to {output_file}")

if __name__ == "__main__":
    main()
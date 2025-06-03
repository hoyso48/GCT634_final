from pydx7 import dx7_synth, midi_note
import numpy as np

def render_from_specs(specs: dict, sr=48000, n=60, v=100, out_scale=1.0):
    synth = dx7_synth(specs, sr=sr, block_size=64) # block_size can be adjusted

    # Create a note sequence: one note, velocity 100, on for 1.0 seconds, off for 0.5 seconds
    # Ensure ton and toff are integers for frame counts
    ton_frames = int(sr * 0.1) 
    toff_frames = int(sr * 0.05)
    note = midi_note(n=n, v=v, ton=ton_frames, toff=toff_frames, silence=0)
    audio = synth.render_from_midi_sequence([note])

    final_audio = (audio * 32767 * out_scale).astype(np.int16)
    return final_audio

def serialize_specs(specs: dict) -> str:
    def serialize_value(v_param):
        value_to_process = v_param
        if isinstance(v_param, np.ndarray):
            value_to_process = v_param.tolist()

        if isinstance(value_to_process, list):
            is_list_of_sublists = False
            if value_to_process:  # Only check non-empty lists
                is_list_of_sublists = any(isinstance(item, list) for item in value_to_process)

            if is_list_of_sublists:
                item_repr_strings = []
                for sub_item in value_to_process:
                    item_repr_strings.append(f"\t\t{repr(sub_item)}")
                
                joined_items_string = ",\n".join(item_repr_strings)
                # Add a newline before the closing bracket if there were items
                closing_bracket_prefix = "\n" if value_to_process else ""
                return f"[\n{joined_items_string}{closing_bracket_prefix}    ]" if value_to_process else "[]" # Handle totally empty list vs list of lists that might be empty
            else:
                # Simple list (e.g., list of numbers, or an empty list [])
                return repr(value_to_process)

        elif isinstance(value_to_process, np.generic): # Handles numpy scalars like np.int64, np.uint8, np.float32 etc.
            return repr(value_to_process.item())
        
        elif isinstance(value_to_process, (int, float, bool, str)): # Standard Python types
            return repr(value_to_process)
        
        elif value_to_process is None:
            return repr(None)
            
        else:
            raise ValueError(f"Unsupported type: {type(value_to_process)} (original type: {type(v_param)}) for value: {value_to_process}")

    lines = ["{"]
    for key, value in specs.items():
        serialized = serialize_value(value)
        lines.append(f"    {repr(key)}: {serialized},")
    lines.append("}")

    return "\n".join(lines)

def validate_specs(specs, syx_file='', patch_number=-1):
    valid = True
    if 'name' not in specs:
        print(f"[WARNING] {syx_file}: patch {patch_number}: 'name' is not in specs")
        patch_name = 'NaN'
    elif specs['name'] == '':
        print(f"[WARNING] {syx_file}: patch {patch_number}: 'name' is empty.")
        patch_name = 'NaN'
    else:
        patch_name = specs['name']
    
    if not isinstance(patch_name, str):
        print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: 'name' is not a string.")
        patch_name = 'NaN'

    def check_range(name, value, lo, hi):
        if not lo <= value <= hi:
            print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}' = {value} is out of range [{lo}, {hi}]")
            valid = False

    def check_list_range(name, lst, lo, hi):
        for idx, v in enumerate(lst):
            if not lo <= v <= hi:
                print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}[{idx}]' = {v} is out of range [{lo}, {hi}]")
                valid = False

    def check_matrix_range(name, matrix, lo, hi):
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                if not lo <= v <= hi:
                    print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: '{name}[{i}][{j}]' = {v} is out of range [{lo}, {hi}]")
                    valid = False

    # Validate all fields
    check_matrix_range("modmatrix", specs['modmatrix'], 0, 1)

    feedback_count = sum([specs['modmatrix'][i][i] for i in range(6)])
    if feedback_count > 1:
        print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: multiple operators have feedback.")
        valid = False

    check_list_range("outmatrix", specs['outmatrix'], 0, 1)
    check_range("feedback", specs['feedback'], 0, 7)
    check_list_range("coarse", specs['coarse'], 0, 31)
    check_list_range("fine", specs['fine'], 0, 99)
    check_list_range("detune", specs['detune'], -7, 7)
    check_range("transpose", specs['transpose'], -24, 24)
    check_list_range("ol", specs['ol'], 0, 99)

    for r in range(4):
        check_list_range(f"eg_rate[{r}]", specs['eg_rate'][r], 0, 99)
        check_list_range(f"eg_level[{r}]", specs['eg_level'][r], 0, 99)

    check_list_range("sensitivity", specs['sensitivity'], 0, 7)

    if not isinstance(specs['has_fixed_freqs'], bool):
        print(f"[WARNING] {syx_file}: patch {patch_number} {patch_name}: 'has_fixed_freqs' is not a boolean.")
        valid = False
    
    return valid
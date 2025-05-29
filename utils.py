from pydx7 import dx7_synth, midi_note
import numpy as np

def render_from_specs(specs: dict, sr=44100, n=64, out_scale=0.75):
    synth = dx7_synth(specs, sr=sr, block_size=64) # block_size can be adjusted

    # Create a note sequence: one note, velocity 100, on for 1.0 seconds, off for 0.5 seconds
    # Ensure ton and toff are integers for frame counts
    sr_val = 44100
    ton_frames = int(sr_val * 0.1) 
    toff_frames = int(sr_val * 0.05)
    note = midi_note(n=n, v=100, ton=ton_frames, toff=toff_frames, silence=0)
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

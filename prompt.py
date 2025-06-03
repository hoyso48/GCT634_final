base_prompt = """Find the parameter values for the famous 6-OP(operator) FM synthesizer, the DX7, such that the resulting sound matches the given prompt.
"""

zeroshot_prompt = base_prompt + """
Make sure to follow the python dictionary format given below.
```python
specs = {
	# parameters are indexed from OP1(index 0) to OP6(index 5)
    'modmatrix': [[int ∈ {0,1}, ...] * 6] * 6, # 6x6 binary matrix. modmatrix[i][j] = 1 means OP(i+1) is modulated by OP(j+1). Diagonal 1 means feedback(at most one OP can have feedback).
    'outmatrix': [int ∈ {0,1}] * 6,            # Binary flags for which operators send output to the final mix (1=active carrier).
    'feedback': int ∈ [0, 7],                  # Feedback intensity for the operator with self-modulation (modmatrix[i][i] = 1).
    'coarse': [int ∈ [0, 31]] * 6,             # Coarse frequency ratios. Integer values determining harmonic relationship.
    'fine': [int ∈ [0, 99]] * 6,               # Fine frequency offset. Adds fractional harmonic variation.
    'detune': [int ∈ [-7, 7]] * 6,             # Pitch offset in cents. -7 ~ +7 range to enrich timbre.
    'transpose': int ∈ [-24, 24],              # Global pitch shift in semitones. 0 = C3.
    'ol': [int ∈ [0, 99]] * 6,                 # Output Level per OP. Controls amplitude of each operator.
    'eg_rate': [
        [int ∈ [0, 99]] * 6,  # Rate 1 (Attack Rate)
        [int ∈ [0, 99]] * 6,  # Rate 2 (Decay 1)
        [int ∈ [0, 99]] * 6,  # Rate 3 (Sustain rate or Decay 2)
        [int ∈ [0, 99]] * 6   # Rate 4 (Release rate)
    ],  # Envelope generator speeds
    'eg_level': [
        [int ∈ [0, 99]] * 6,  # Level 1 (Attack Level)
        [int ∈ [0, 99]] * 6,  # Level 2 (Decay Level)
        [int ∈ [0, 99]] * 6,  # Level 3 (Sustain Level)
        [int ∈ [0, 99]] * 6   # Level 4 (Release Level)
    ],  # Envelope generator target levels
    'sensitivity': [int ∈ [0, 7]] * 6,         # Velocity sensitivity per operator. 0 (none) to 7 (max)
    'has_fixed_freqs': bool                   # True: fixed frequency mode per operator (ignores pitch), False: follows key pitch
}```
### Prompt: {}
"""

example_zeroshot = zeroshot_prompt.format("Bells, which have a bright, metallic timbre with a quick, percussive attack and a long, resonant decay.")
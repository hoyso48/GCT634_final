base_prompt = """Find the parameter values for the famous 6-OP(operator) FM synthesizer, the DX7, such that the resulting sound matches the given prompt.
"""

zeroshot_prompt = base_prompt + """
Make sure to follow the python dictionary format given below.
```python
specs = {{
	# parameters are indexed from OP1(index 0) to OP6(index 5)
    'modmatrix': [[int ∈ {{0,1}}, ...] * 6] * 6, # 6x6 binary matrix. modmatrix[i][j] = 1 means OP(i+1) is modulated by OP(j+1). Diagonal 1 means feedback(at most one OP can have feedback).
    'outmatrix': [int ∈ {{0,1}}] * 6,            # Binary flags for which operators send output to the final mix (1=active carrier).
    'feedback': int ∈ [0, 7],                  # Feedback intensity for the operator with self-modulation (modmatrix[i][i] = 1).
    'fixed_freq': [int ∈ [0, 1]] * 6,          # Binary flags for which operators have a fixed frequency(not affected by key). when set to 1, operator's frequency is fixed to 10**(coarse%4 + fine/10). detune is ignored.
    'coarse': [int ∈ [0, 31]] * 6,             # Coarse frequency ratios. Integer values determining harmonic relationship. 0 means 0.5 particularly.
    'fine': [int ∈ [0, 99]] * 6,               # Fine frequency offset. Adds fractional harmonic variation. 
    'detune': [int ∈ [-7, 7]] * 6,             # Pitch offset in cents. -7 ~ +7 range to enrich timbre.
    'transpose': int ∈ [-24, 24],              # Global pitch shift in semitones.
    'ol': [int ∈ [0, 99]] * 6,                 # Output Level per OP. Controls amplitude of each operator.
    'eg_rate': [
        [int ∈ [0, 99]] * 6,  # Rate 1 (Attack Rate)
        [int ∈ [0, 99]] * 6,  # Rate 2 (Decay 1)
        [int ∈ [0, 99]] * 6,  # Rate 3 (Sustain rate or Decay 2)
        [int ∈ [0, 99]] * 6   # Rate 4 (Release rate)
    ],  # Envelope generator speeds, high rate means faster.
    'eg_level': [
        [int ∈ [0, 99]] * 6,  # Level 1 (Attack Level)
        [int ∈ [0, 99]] * 6,  # Level 2 (Decay Level)
        [int ∈ [0, 99]] * 6,  # Level 3 (Sustain Level)
        [int ∈ [0, 99]] * 6   # Level 4 (Release Level)
    ],  # Envelope generator target levels, high level means louder.
    'sensitivity': [int ∈ [0, 7]] * 6,         # Velocity sensitivity per operator. 0 (none) to 7 (max)
}}```
### Prompt: {prompt}"""

zeroshot_prompt = base_prompt + "### Prompt: {prompt}"

# example_zeroshot = zeroshot_prompt + "Bells, which have a bright, metallic timbre with a quick, percussive attack and a long, resonant decay."



# zeroshot_prompt = base_prompt + """
# Make sure to follow the python dictionary format given below.
# ```python
# specs = {{
# 	# parameters are indexed from OP1(index 0) to OP6(index 5)
#     # 6x6 binary matrix. modmatrix[i][j] = 1 means OP(i+1) is modulated by OP(j+1). Diagonal 1 means feedback(at most one OP can have feedback).
#     'modmatrix': [[int ∈ {{0,1}}, ...] * 6] * 6,
#     # Binary flags for which operators send output to the final mix (1=active carrier).
#     'outmatrix': [int ∈ {{0,1}}] * 6,
#     # Feedback intensity for the operator with self-modulation (modmatrix[i][i] = 1).
#     'feedback': int ∈ [0, 7],
#     # Coarse frequency ratios. Integer values determining harmonic relationship.
#     'coarse': [int ∈ [0, 31]] * 6,
#     # Fine frequency offset. Adds fractional harmonic variation.
#     'fine': [int ∈ [0, 99]] * 6,
#     # Pitch offset in cents. -7 ~ +7 range to enrich timbre.
#     'detune': [int ∈ [-7, 7]] * 6,
#     # Global pitch shift in semitones. 0 = C3.
#     'transpose': int ∈ [-24, 24],
#     # Output Level per OP. Controls amplitude of each operator.
#     'ol': [int ∈ [0, 99]] * 6,
#     # Envelope generator speeds
#     'eg_rate': [
#         # Rate 1 (Attack Rate)
#         [int ∈ [0, 99]] * 6, 
#         # Rate 2 (Decay 1)
#         [int ∈ [0, 99]] * 6,
#         # Rate 3 (Sustain rate or Decay 2)
#         [int ∈ [0, 99]] * 6,
#         # Rate 4 (Release rate)
#         [int ∈ [0, 99]] * 6   
#     ],
#     # Envelope generator target levels
#     'eg_level': [
#         # Level 1 (Attack Level)
#         [int ∈ [0, 99]] * 6,
#         # Level 2 (Decay Level)
#         [int ∈ [0, 99]] * 6,
#         # Level 3 (Sustain Level)
#         [int ∈ [0, 99]] * 6,
#         # Level 4 (Release Level)
#         [int ∈ [0, 99]] * 6
#     ],
#     # Velocity sensitivity per operator. 0 (none) to 7 (max)
#     'sensitivity': [int ∈ [0, 7]] * 6,
# }}```
# ### Prompt: {prompt}"""
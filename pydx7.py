import numpy as np
from numba import njit
from einops import rearrange
import copy # For deepcopy

LG_N = 6

MOD_MATRIX = []

# alg 1       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,1] ])    
# alg 2       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,1,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
# alg 3       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,1] ])
# alg 4       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
# alg 5       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1] ])
# alg 6       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0] ])

# alg 7       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1] ])
# alg 8       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,0], [0,0,0,1,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
# alg 9       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,1,0,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

# alg 10       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
# alg 11       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,1] ])

# alg 12       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,1,0,0,0,0], [0,0,0,1,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
# alg 13       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1] ])

# alg 14       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,1] ])
# alg 15       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,1,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

# alg 16       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,1,0,1,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1] ])
# alg 17       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,1,0,1,0], [0,1,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

# alg 18       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,1,1,0,0], [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

# alg 19       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1] ])

# alg 20       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

# alg 21       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

# alg 22       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1] ])

# alg 23       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1] ])

# alg 24       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1] ])

# alg 25       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1] ])

# alg 26       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,1] ])
# alg 27       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

# alg 28       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,1,0], [0,0,0,0,0,0] ])

# alg 29       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1] ])

# alg 30       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,1,0], [0,0,0,0,0,0] ])

# alg 31       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1] ])

# alg 32       OP1            OP2            OP3            OP4            OP5            OP6
MOD_MATRIX.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1] ])

MOD_MATRIX = np.array(MOD_MATRIX)

OUT_MATRIX = [
        [1,0,1,0,0,0], #1
        [1,0,1,0,0,0], #2
        [1,0,0,1,0,0], #3
        [1,0,0,1,0,0], #4
        [1,0,1,0,1,0], #5
        [1,0,1,0,1,0], #6
        [1,0,1,0,0,0], #7
        [1,0,1,0,0,0], #8
        [1,0,1,0,0,0], #9
        [1,0,0,1,0,0], #10

        [1,0,0,1,0,0], #11
        [1,0,1,0,0,0], #12
        [1,0,1,0,0,0], #13
        [1,0,1,0,0,0], #14
        [1,0,1,0,0,0], #15
        [1,0,0,0,0,0], #16
        [1,0,0,0,0,0], #17
        [1,0,0,0,0,0], #18
        [1,0,0,1,1,0], #19
        [1,1,0,1,0,0], #20

        [1,1,0,1,1,0], #21
        [1,0,1,1,1,0], #22
        [1,1,0,1,1,0], #23
        [1,1,1,1,1,0], #24
        [1,1,1,1,1,0], #25
        [1,1,0,1,0,0], #26
        [1,1,0,1,0,0], #27
        [1,0,1,0,0,1], #28
        [1,1,1,0,1,0], #29
        [1,1,1,0,0,1], #30

        [1,1,1,1,1,0], #31
        [1,1,1,1,1,1], #32
    ]
    
OUT_MATRIX = np.array(OUT_MATRIX)

def get_modmatrix(algorithm: int) -> np.array:
    """
        Returns the modulation matrix corresponding to the algorithm selected.
    """
    return MOD_MATRIX[algorithm]

def get_outmatrix(algorithm: int) -> np.array:
    return OUT_MATRIX[algorithm]

def get_algorithm(mod_matrix:np.array,out_matrix:np.array) -> int:
    for i in range(len(MOD_MATRIX)):
        if(np.array_equal(MOD_MATRIX[i],mod_matrix) and np.array_equal(OUT_MATRIX[i],out_matrix)):
            return i
    print("Algorithm not found")
    return -1

# Loads the first patch

def load_patch_from_bulk(patch_file,patch_number:int = 0,load_from_sysex=False):
    '''
    TODO: Incorporate:  
        - KB RATE Scaling (from dx7 users manual: "The EG for each operator can be set for a
                        long bass decay and a short treble decay - as in an acoustic piano")
        - OP Detune parameter
    
    Args:
        patch_file: Path to dx7 cart file
        patch_number: Position of patch within cart
        load_from_sysex: Set it to 'True' when cart file is a sysex dump
    '''
    bulk_patches = np.fromfile(patch_file, dtype=np.uint8)

    patch_offset = 6 if load_from_sysex==True else 0
    for i in [patch_number]:
        patch = bulk_patches[patch_offset + i*128:patch_offset+ (i+1)*128]

    return load_patch(patch)

def load_patch(patch : np.array):
    '''
    Unpacks patch array from cart file and generates a patch structure (called here 'spec')
    '''
    specs = {}
    # Store binary data from patch.
    # specs['binary'] = patch

    patch_name = patch[118:127]
    patch_name = patch_name * ( patch_name < 128)
    specs['name'] = patch_name.tostring().decode('ascii')

    patch = unpack_packed_patch(patch)
    algorithm = patch[134]

    # fr = np.zeros(6,dtype=float)
    coares = np.zeros(6,dtype=int)
    fine = np.zeros(6,dtype=int)
    detune = np.zeros(6,dtype=int)
    ol = np.zeros(6,dtype=int)
    rates = np.zeros([4,6],dtype=int)
    levels = np.zeros([4,6],dtype=int)
    sensitivity = np.zeros(6,dtype=int)
    fixed_freq = np.zeros(6,dtype=int)

    # https://homepages.abdn.ac.uk/d.j.benson/pages/dx7/sysex-format.txt
    # Load OP output level, EG rates and levels
    has_fixed_freq = False
    for op in range(6):
        # First in file is OP6
        off = op*21
        is_fixed = patch[off+17]
        fixed_freq[5-op] = is_fixed
        if(is_fixed):
            #print("[WARNING] tools.py: OP{} in {} is FIXED.".format(6-op,patch_name))
            has_fixed_freqs = True

        ol[5-op] = patch[off+16]
        sensitivity[5-op] = patch[off+15]
        
        #compute frequency value    
        f_coarse = patch[off+18]
        f_fine = patch[off+19]
        f_detune = patch[off+20]
        coares[5-op] = f_coarse
        fine[5-op] = f_fine
        # detune is -7~+7, where 0 means no detune.
        detune[5-op] = f_detune - 7
        # fr[5-op] = compute_freq(f_coarse,f_fine,f_detune) #Detune is ignored for now.
        
        # get rates and levels
        for i in range(4):
            rates[i,5-op] = patch[off+i]
            levels[i,5-op] = patch[off+4+i]

    transpose = (patch[144]-24)
    # print(np.where(patch == 6))
    feedback = patch[135]
    # print(transpose)
    # factor = 2**(transpose/12)
    # fr = factor*fr
    
    # specs['fr'] = fr
    specs['modmatrix'] = get_modmatrix(algorithm)
    specs['outmatrix'] = get_outmatrix(algorithm)
    specs['feedback'] = feedback
    specs['fixed_freq'] = fixed_freq[::-1]
    specs['coarse'] = coares[::-1]
    specs['fine'] = fine[::-1]
    specs['detune'] = detune[::-1]
    specs['transpose'] = transpose
    specs['ol'] = ol[::-1]
    specs['eg_rate'] = rates[:,::-1]
    specs['eg_level'] = levels[:,::-1]
    specs['sensitivity'] = sensitivity[::-1]
    # specs['algorithm'] = algorithm #0-31
    specs['has_fixed_freqs'] = has_fixed_freqs
    #normalize coarse and detune for fixed freqs
    for i in range(6):
        if specs['fixed_freq'][i] == 1:
            specs['coarse'][i] = specs['coarse'][i] % 4
            specs['detune'][i] = 0
    return specs

def compute_freq(coarse,fine,detune):
    '''
    Converts from DX7 format to a floating point frequency ratio.
    '''
    # TODO detune parameter is -7 to 7 cents (not implemented)
    f = coarse
    if (f==0): f = 0.5
    f = f + (f/100)*fine
    return f

# Nice unpacking method adapted from https://github.com/bwhitman/learnfm
def unpack_packed_patch(p):
    ''' 
    p: is a 128 byte array extracted from the DX7 cart.
    Returns:
        a parsed 156 byte array that can be easily processed.
    '''
    o = [0]*156
    for op in range(6):
        o[op*21:op*21 + 11] = p[op*17:op*17+11]
        leftrightcurves = p[op*17+11]
        o[op * 21 + 11] = leftrightcurves & 3
        o[op * 21 + 12] = (leftrightcurves >> 2) & 3
        detune_rs = p[op * 17 + 12]
        o[op * 21 + 13] = detune_rs & 7
        o[op * 21 + 20] = detune_rs >> 3
        kvs_ams = p[op * 17 + 13]
        o[op * 21 + 14] = kvs_ams & 3
        o[op * 21 + 15] = kvs_ams >> 2
        o[op * 21 + 16] = p[op * 17 + 14]
        fcoarse_mode = p[op * 17 + 15]
        o[op * 21 + 17] = fcoarse_mode & 1
        o[op * 21 + 18] = fcoarse_mode >> 1
        o[op * 21 + 19] = p[op * 17 + 16]

    o[126:126+9] = p[102:102+9]
    oks_fb = p[111]
    # o[111] = oks_fb
    o[135] = oks_fb & 7
    o[136] = oks_fb >> 3
    o[137:137+4] = p[112:112+4]
    lpms_lfw_lks = p[116]
    o[141] = lpms_lfw_lks & 1
    o[142] = (lpms_lfw_lks >> 1) & 7
    o[143] = lpms_lfw_lks >> 4
    o[144:144+11] = p[117:117+11]
    o[155] = 0x3f #Seems that OP ON/OFF they are always on. Ignore.

    # Clamp the unpacked patches to a known max. 
    maxes =  [
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc6
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc5
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc4
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc3
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc2
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc1
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, # pitch eg rate & level 
        31, 7, 1, 99, 99, 99, 99, 1, 5, 7, 48, # algorithm etc
        126, 126, 126, 126, 126, 126, 126, 126, 126, 126, # name
        127 # operator on/off
    ]
    for i in range(156):
        if(o[i] > maxes[i]): o[i] = maxes[i]
        if(o[i] < 0): o[i] = 0
    return o

#See "velocity" section of notes of dexed. Returns velocity delta in microsteps.
def scalevelocity(velocity:int, sensitivity:int):
    velocity_data = [
      0, 70, 86, 97, 106, 114, 121, 126, 132, 138, 142, 148, 152, 156, 160, 163,
      166, 170, 173, 174, 178, 181, 184, 186, 189, 190, 194, 196, 198, 200, 202,
      205, 206, 209, 211, 214, 216, 218, 220, 222, 224, 225, 227, 229, 230, 232,
      233, 235, 237, 238, 240, 241, 242, 243, 244, 246, 246, 248, 249, 250, 251,
      252, 253, 254]
    clamped_vel = max(0, min(127, velocity))
    vel_value = velocity_data[clamped_vel >> 1] - 239
    scaled_vel = ((sensitivity * vel_value + 7) >> 3) << 4
    return scaled_vel

def scaleoutlevel(outlevel:int):
    levellut = [ 0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46]
    return 28 + outlevel if outlevel >= 20 else levellut[outlevel]

class EnvelopeGenerator():
    '''
    Yamaha DX7 ADSR EG reimplementation.
    Adapted from the C++ implementation of Dexed
    
    '''
    def __init__(self,r:np.ndarray,l:np.ndarray,ol): # Changed type hint for r, l to ndarray
        # 0:Attack,1:Release,2:Sustain,3:Decay
        self.ix = 0
        self.rates = r
        self.levels = l
        self.outlevel = ol
        #self.rate_scaling #TODO
        self.level = 0
        self.targetlevel = 0
        self.inc = 0
        self.rising = False
        self.down = True
        self.advance(0)

    # call env.keydown(false) to release note.
    def keydown(self,d:bool):
        if (self.down != d):
            self.down = d
            if(d):
                self.advance(0)
            else:
                self.advance(3)

    def setparam(self,param:int, value:int):
        if (param < 4):
            self.rates[param] = value
        elif (param < 8):
            self.levels[param - 4] = value

    def advance(self,newix:int):
        self.ix = newix
        if (self.ix < 4):
            newlevel = self.levels[self.ix];
            #/*Pass from 0-99 to 0-127, then to the level of 64 values?*/
            actuallevel = scaleoutlevel(newlevel) >> 1
            #/*Multiply (in log space . . .) the op outlevel with the level of the EG.*/
            actuallevel = (actuallevel << 6) + self.outlevel - 4256

            #/*Set a minimum possible level.*/
            actuallevel = 16 if (actuallevel < 16 ) else actuallevel
            self.targetlevel = actuallevel << 16
            self.rising = (self.targetlevel > self.level)

            #// rate

            #/* max val: 99*41 = 4059 - > turn from 12 to 6 bits. */
            qrate = (self.rates[self.ix] * 41) >> 6
            #/*(in log space) multiply rate by the rate scaling.*/
            
            # Rate scaling not applied
            #qrate += self.rate_scaling
            
            qrate = min(qrate, 63)
            self.inc = (4 + (qrate & 3)) << (2 + LG_N + (qrate >> 2))


    def getsample(self):
        '''
        Result is in Q24/doubling log (log2) format. Also, result is subsampled
        for every N samples.
        A couple more things need to happen for this to be used as a gain
        value. First, the # of outputs scaling needs to be applied. Also,
        modulation.
        Then, of course, log to linear.
        '''
        if (self.ix < 3 or ((self.ix < 4) and (not self.down))):
            if (self.rising):
                jumptarget = 1716
                if (self.level < (jumptarget << 16)): 
                    self.level = jumptarget << 16
                self.level += (((17 << 24) - self.level) >> 24) * self.inc
                #print(" self.level {}".format(self.level))
                #// TODO: should probably be more accurate when inc is large
                if (self.level >= self.targetlevel):
                    self.level = self.targetlevel
                    self.advance(self.ix + 1)
            else:
                #!rising
                self.level -= self.inc
                if (self.level <= self.targetlevel):
                    self.level = self.targetlevel
                    self.advance(self.ix + 1)
        return self.level

def render_env(rate,level,ol:int,sens,velocity:int,frames_on:int,
               frames_off:int,qenvelopes_ratio:float=1.0):
    output_level = scaleoutlevel(ol)
    output_level = output_level << 5
    output_level += scalevelocity(velocity, sens)
    output_level = max(0, output_level)

    e= EnvelopeGenerator(rate,level,output_level)

    e.keydown(True)
    n_frames = frames_on + frames_off
    gain = np.zeros(n_frames,dtype=float)
    qgain = np.zeros(n_frames,dtype=float)
    for i in range(n_frames):
        out = e.getsample()
        qgain[i] = out*qenvelopes_ratio
        a = 2**(10 + qgain[i] * ( 1.0/(1<<24) ) )/(1<<24)
        gain[i] = a

        if i == (frames_on):
            e.keydown(False)
    return [gain,qgain]

def render_envelopes(specs,velocity,frames_on,frames_off,
    qenvelopes_ratio=None): # Default to None
    if qenvelopes_ratio is None: # Provide default if not given
        qenvelopes_ratio=[1.0,1.0,1.0,1.0,1.0,1.0]
        
    envelopes = np.zeros([6,frames_on + frames_off])
    qenvelopes = np.zeros([6,frames_on + frames_off])
    for i in range(6):
        out = render_env(specs['eg_rate'][:,5-i],
                                specs['eg_level'][:,5-i],
                                specs['ol'][5-i],
                                specs['sensitivity'][5-i],
                                velocity,frames_on,frames_off,
                                qenvelopes_ratio[i])
        envelopes[i,:] = out[0]
        qenvelopes[i,:] = out[1]

    return [envelopes,qenvelopes]

def upsample(signal, factor):
    n = signal.shape[0]
    x = np.linspace(0,n-1,n)
    xvals = np.linspace(0,n-1,int(n*factor)) # Ensure factor is int for array size
    if (len(signal.shape) == 2):
      interpolated = np.zeros((int(n*factor),signal.shape[1]))
      for i in range(signal.shape[1]):
        interpolated[:,i] = np.interp(xvals,x,signal[:,i])
    else:
      interpolated = np.interp(xvals,x,signal)
    return interpolated

def freq_not_fixed(coarse_vals, fine_vals, detune_vals, transpose):
    # Calculate base frequency ratio from coarse and fine parameters
    # Where coarse is 0, use 0.5, otherwise use the coarse value.
    f_ratio_vals = np.where(coarse_vals == 0, 0.5, coarse_vals)
    
    # Apply fine tune: fine is 0~99.
    # f_new = f_old * (1 + fine/100)
    f_ratio_vals = f_ratio_vals * (1.0 + fine_vals / 100.0)

    # Apply detune: detune is -7~+7
    # Calculate the frequency multiplier from cents.
    # Formula: multiplier = 2^(cents / 1200)
    detune_multiplier_vals = 2.0**(detune_vals / 1200.0)
    
    final_freq_ratio_vals = f_ratio_vals * detune_multiplier_vals

    #apply transpose
    factor = 2**(transpose/12)
    final_freq_ratio_vals = factor * final_freq_ratio_vals

    return final_freq_ratio_vals

def freq_fixed(coarse_vals, fine_vals, detune_vals, transpose):
    power_of_10 = coarse_vals % 4 
    base_freq_from_coarse_range = 10.0 ** power_of_10
    fine_tune_multiplier = 1.0 + 0.08861 * fine_vals
    freq= base_freq_from_coarse_range * fine_tune_multiplier
    return freq

def compute_freq(coarse_arr, fine_arr, detune_arr, transpose, fixed_freq):
    '''
    Converts from DX7 format parameters to floating point frequency ratios, including detune.
    Operates element-wise on input NumPy arrays.

    Args:
        coarse_arr (np.ndarray): Array of coarse frequency parameters (shape (6,)).
        fine_arr (np.ndarray): Array of fine frequency parameters (shape (6,)).
        detune_arr (np.ndarray): Array of DX7 operator detune parameters (shape (6,)).
                                 Each element is -7~+7, where 0 means no detune.
                                 A difference of 1 corresponds to 1 cent.
        transpose (int): Transposition in semitones.

    Returns:
        np.ndarray: Array of computed frequency ratios (shape (6,)).
    '''
    # Convert inputs to float arrays for calculations
    coarse_vals = coarse_arr.astype(float)
    fine_vals = fine_arr.astype(float)
    detune_vals = detune_arr.astype(float)
    fixed_freq = fixed_freq.astype(bool)

    final_freq = np.where(fixed_freq, freq_fixed(coarse_vals, fine_vals, detune_vals, transpose), freq_not_fixed(coarse_vals, fine_vals, detune_vals, transpose))
    return final_freq
    
@njit
def dx7_numba_render(fr: np.ndarray, modmatrix: np.ndarray, outmatrix: np.ndarray,
                     pitch: np.ndarray, ol: np.ndarray, sr: int, fixed_freq: np.ndarray,
                     feedback_level_param: int, scale: float = 2 * np.pi):
    """
    6-operator FM Renderer with numba, including feedback loop.
    """
    n_op = len(fr)
    out = np.zeros_like(pitch)
    phases = np.zeros(n_op)  # The free-running phase
    tstep = 1.0 / sr

    # Store previous two outputs of operators for feedback: [op_idx, 0: s-1, 1: s-2]
    prev_op_outputs = np.zeros((n_op, 2))

    # # Feedback scaling factors.
    # # fb_factors_arr[0] is for feedback_level 0, fb_factors_arr[1] for level 1, etc.
    # # Level 0: 0 (no feedback)
    # # Level 1: 2^-7 (1/128)
    # # Level 7: 2^-1 (1/2)
    # feedback_factors_arr = np.array([0.0] + [2.0**(level - 8) for level in range(0, 7)]) # Corrected range for 0-7 to map to 1-7 index
    # # For feedback_level_param from 0 to 7:
    # # feedback_factors_arr[0] is 0.0 (for level 0)
    # # feedback_factors_arr[1] is 2.0**(-7) (for level 1) ... feedback_factors_arr[7] is 2.0**(-1) (for level 7)
    
    # # Regenerate with correct indexing for feedback_level_param (0-7)
    # _fb_factors_temp = [0.0] # for level 0
    # for i in range(1, 8): # for levels 1 to 7
    #     _fb_factors_temp.append(2.0**(i - 8))
    # feedback_factors_arr = np.array(_fb_factors_temp)

    # current_fb_factor = 0.0
    # if 0 <= feedback_level_param < len(feedback_factors_arr):
    #     current_fb_factor = feedback_factors_arr[feedback_level_param]

    for s in range(out.shape[0]):
        # Update base phases for all operators
        for op_idx in range(n_op):
            if fixed_freq[op_idx] == 1:
                phases[op_idx] += tstep * 2 * np.pi * fr[op_idx]
            else:
                phases[op_idx] += tstep * 2 * np.pi * pitch[s] * fr[op_idx]
            # Ensure phase remains within [0, 2*pi) - Numba friendly
            while phases[op_idx] >= 2 * np.pi:
                phases[op_idx] -= 2 * np.pi
            while phases[op_idx] < 0: # Should not happen with positive tstep/pitch/fr but good practice
                phases[op_idx] += 2 * np.pi
        
        # Copy base phases to modifiable phases for this sample
        current_sample_modphases = phases.copy()

        # 1. Apply feedback modulation to operators that have it
        if feedback_level_param > 0: # Only if feedback is active globally
            for op_fb_idx in range(n_op):
                if modmatrix[op_fb_idx, op_fb_idx] == 1: # Check if this operator has feedback
                    avg_prev_out = (prev_op_outputs[op_fb_idx, 0] + prev_op_outputs[op_fb_idx, 1]) / 2.0
                    feedback_phase_mod = (avg_prev_out / 2**(8-feedback_level_param)) * scale
                    current_sample_modphases[op_fb_idx] += feedback_phase_mod

        # 2. Apply inter-operator modulation
        for mod_op in range(n_op - 1, -1, -1):
            modulator_output_for_pm = np.sin(current_sample_modphases[mod_op]) * ol[s, mod_op] * scale
            for carr_op in range(n_op):
                if modmatrix[carr_op, mod_op] == 1 and carr_op != mod_op:
                    current_sample_modphases[carr_op] += modulator_output_for_pm
        
        # 3. Calculate final audio output of each operator for this sample
        #    and update history for next sample's feedback calculation.
        actual_op_audio_outputs_this_sample = np.zeros(n_op)
        for op_idx in range(n_op):
            op_output_value = np.sin(current_sample_modphases[op_idx]) * ol[s, op_idx]
            actual_op_audio_outputs_this_sample[op_idx] = op_output_value
            
            # Update history for feedback: shift s-1 to s-2, store current (s) as new s-1
            # This is done for all operators; feedback loop will pick the correct one if active.
            prev_op_outputs[op_idx, 1] = prev_op_outputs[op_idx, 0]
            prev_op_outputs[op_idx, 0] = op_output_value
            
        # 4. Sum outputs of carrier operators
        out[s] = np.sum(outmatrix * actual_op_audio_outputs_this_sample)

    return out

class midi_note():
  def __init__(self,n:int=0,v:int=0,ton:int=0,toff:int=0,silence:int=0):
    self.n = n
    self.v = v
    self.ton = ton
    self.toff = toff
    self.silence = silence

class dx7_synth():
  def __init__(self,specs,sr:int=44100,block_size:int=64):
    self.specs = specs
    self.modmatrix = np.array(specs['modmatrix']) # Ensure numpy array
    self.outmatrix = np.array(specs['outmatrix']) # Ensure numpy array
    self.fr = compute_freq(np.array(specs['coarse'][::-1]),np.array(specs['fine'][::-1]),np.array(specs['detune'][::-1]),np.array(specs['transpose']),np.array(specs['fixed_freq'][::-1]))
    self.scale = 2*np.pi # Default modulation index scale
    self.sr = sr
    self.block_size = block_size
    self.fixed_freq = np.array(specs['fixed_freq'][::-1])
    # Store feedback level from specs, default to 0 if not present
    self.feedback_level = specs.get('feedback', 0) 

  def render_from_osc_envelopes(self,f0: np.ndarray,ol: np.ndarray):
    ol_up = upsample(ol,self.block_size)
    f0_up = upsample(f0,self.block_size)
    
    render = dx7_numba_render(self.fr, self.modmatrix, self.outmatrix,
                    f0_up, ol_up, self.sr, self.fixed_freq, self.feedback_level, self.scale)
    
    num_carriers = np.sum(self.outmatrix)
    if num_carriers == 0: 
        num_carriers = 1 # Avoid division by zero
    
    # Original normalization factor
    return render / (4.0 * num_carriers)


  def render_from_midi_sequence(self,midi_sequence):
    envelopes = np.empty((6,0))
    note_contour = np.empty(0)
    
    for entry in midi_sequence:
      if(entry.silence == 0):
        # Ensure eg_rate and eg_level are numpy arrays before passing
        current_specs = self.specs.copy() # To avoid modifying the original dict
        current_specs['eg_rate'] = np.array(self.specs['eg_rate'])
        current_specs['eg_level'] = np.array(self.specs['eg_level'])

        env,qenv = render_envelopes(current_specs,entry.v,entry.ton,entry.toff)
        envelopes = np.append(envelopes,env,axis=1)
        note_contour = np.append(note_contour,np.ones(entry.ton+entry.toff)*entry.n)
      else:
        envelopes = np.append(envelopes,np.zeros((6,entry.silence)),axis=1)
        note_contour = np.append(note_contour,np.zeros(entry.silence))
    
    f0 = 440*2**((note_contour-69)/12)
    f0[note_contour == 0] = 0 # Handle silence to avoid errors with 2**(0-69)/12 for silence
    
    envelopes = rearrange(envelopes,"oscillators frames -> frames oscillators")
    audio = self.render_from_osc_envelopes(f0,envelopes)

    return audio


DEFAULT_SPEC_PARAMS = {
    'name': "INITVOICE ", # 10 chars
    'algorithm': 31, # Algorithm 32 (0-indexed default)

    # Operator params (OP1 to OP6 order in lists/arrays)
    # eg_rate: [4 rates][6 ops]
    'eg_rate':    [[95,95,95,95,95,95], [95,95,95,95,95,95], [95,95,95,95,95,95], [95,95,95,95,95,95]],
    # eg_level: [4 levels][6 ops]
    'eg_level':   [[99,99,99,99,99,99], [99,99,99,99,99,99], [99,99,99,99,99,99], [0,0,0,0,0,0]],
    # ol: [6 ops]
    'ol':         [99,0,0,0,0,0], # OP1 (carrier in alg 32) at 99, others 0
    # sensitivity (KVS): [6 ops]
    'sensitivity': [2,0,0,0,0,0],
    # fixed_freq: [6 ops]
    'fixed_freq': [0,0,0,0,0,0],
    # coarse: [6 ops]
    'coarse':     [1,1,1,1,1,1],
    # fine: [6 ops]
    'fine':       [0,0,0,0,0,0],
    # detune (spec format: -7 to +7): [6 ops]
    'detune':     [0,0,0,0,0,0],

    # These are not typically in user spec, but needed for packing.
    # Values from DX7 initialized voice or common defaults.
    # [6 ops]
    'op_kbd_scale_break_pt': [0x27]*6, # C3 (Value 39)
    'op_kbd_scale_l_depth':  [0]*6,
    'op_kbd_scale_r_depth':  [0]*6,
    'op_kbd_scale_l_curve':  [0]*6, # -LIN
    'op_kbd_scale_r_curve':  [0]*6, # -LIN
    'op_kbd_rate_scale':     [0]*6,
    'op_amp_mod_sens':       [0]*6,

    # Pitch EG: [4 rates/levels]
    'pitch_eg_rate':         [99,99,99,99],
    'pitch_eg_level':        [50,50,50,50],

    'feedback':              0,
    'osc_key_sync':          1, # ON in DX7 init
    'lfo_speed':             35,
    'lfo_delay':             0,
    'lfo_pitch_mod_depth':   0,
    'lfo_amp_mod_depth':     0,
    'lfo_key_sync':          1, # ON in DX7 init
    'lfo_waveform':          0, # Triangle
    'lfo_pitch_mod_sens':    0,
    'transpose':             0, # Spec transpose (-24 to +24 for example)
                               # Will be converted to 0-48 for packing (val + 24)
    # modmatrix and outmatrix are derived from algorithm if not provided.
    # If algorithm is not provided, it's derived from modmatrix/outmatrix.
}

def _get_resolved_spec(user_spec, global_defaults):
    """Merges user_spec with global_defaults to create a complete spec."""
    resolved = copy.deepcopy(global_defaults)

    for key, value in user_spec.items():
        if key in ['eg_rate', 'eg_level'] and isinstance(value, list) and \
           isinstance(resolved[key], list):
            # Ensure correct dimensions for array-like params
            # Assuming spec gives [param_idx][op_idx]
            # resolved[key] is already in this format from defaults
            for i in range(len(value)): # Iterate over rates/levels
                if i < len(resolved[key]):
                    for j in range(len(value[i])): # Iterate over ops
                        if j < len(resolved[key][i]):
                             resolved[key][i][j] = value[i][j]
        elif isinstance(value, list) and isinstance(resolved.get(key), list):
             # For 1D lists like 'ol', 'coarse', etc.
            for i in range(len(value)):
                if i < len(resolved[key]):
                    resolved[key][i] = value[i]
        else:
            resolved[key] = value

    # Derive algorithm if not present or if modmatrix/outmatrix are primary
    if 'algorithm' not in user_spec or user_spec.get('algorithm') is None:
        if 'modmatrix' in resolved and 'outmatrix' in resolved:
            alg_num = get_algorithm(np.array(resolved['modmatrix']), np.array(resolved['outmatrix']))
            if alg_num != -1:
                resolved['algorithm'] = alg_num
            # If alg_num is -1, it will use the default algorithm from global_defaults
    elif 'algorithm' in resolved: # User provided algorithm, ensure mod/out matrices match
        alg_num = resolved['algorithm']
        if 0 <= alg_num < len(MOD_MATRIX):
            resolved['modmatrix'] = MOD_MATRIX[alg_num].tolist()
            resolved['outmatrix'] = OUT_MATRIX[alg_num].tolist()
        else: # Invalid algorithm number, fall back to default algorithm's matrices
            default_alg = global_defaults.get('algorithm', 0)
            resolved['modmatrix'] = MOD_MATRIX[default_alg].tolist()
            resolved['outmatrix'] = OUT_MATRIX[default_alg].tolist()
            resolved['algorithm'] = default_alg


    # Ensure name is 10 chars
    resolved['name'] = resolved.get('name', "UNNAMED   ")[:10].ljust(10)

    return resolved

def _pack_voice_to_128_bytes(spec):
    """Packs a single resolved spec dictionary into a 128-byte array."""
    packed_data = bytearray(128)

    # Operator parameters (OP6 down to OP1 for packing)
    # spec stores OP1 to OP6, so op_spec_idx = 5 - op_packed_idx
    for op_packed_idx in range(6): # 0=OP6, 1=OP5, ..., 5=OP1
        op_spec_idx = 5 - op_packed_idx # 0=OP1, ..., 5=OP6 from spec perspective
        base = op_packed_idx * 17

        # EG Rates R1-R4 (Bytes 0-3 for current op block)
        packed_data[base + 0] = int(spec['eg_rate'][0][op_spec_idx]) & 0x7F
        packed_data[base + 1] = int(spec['eg_rate'][1][op_spec_idx]) & 0x7F
        packed_data[base + 2] = int(spec['eg_rate'][2][op_spec_idx]) & 0x7F
        packed_data[base + 3] = int(spec['eg_rate'][3][op_spec_idx]) & 0x7F

        # EG Levels L1-L4 (Bytes 4-7)
        packed_data[base + 4] = int(spec['eg_level'][0][op_spec_idx]) & 0x7F
        packed_data[base + 5] = int(spec['eg_level'][1][op_spec_idx]) & 0x7F
        packed_data[base + 6] = int(spec['eg_level'][2][op_spec_idx]) & 0x7F
        packed_data[base + 7] = int(spec['eg_level'][3][op_spec_idx]) & 0x7F

        # KBD Level Scaling Break Point (Byte 8)
        packed_data[base + 8] = int(spec['op_kbd_scale_break_pt'][op_spec_idx]) & 0x7F
        # KBD Level Scaling Left Depth (Byte 9)
        packed_data[base + 9] = int(spec['op_kbd_scale_l_depth'][op_spec_idx]) & 0x7F
        # KBD Level Scaling Right Depth (Byte 10)
        packed_data[base + 10] = int(spec['op_kbd_scale_r_depth'][op_spec_idx]) & 0x7F

        # Byte 11: KBD L Curve (0-3), R Curve (0-3)
        l_curve = int(spec['op_kbd_scale_l_curve'][op_spec_idx]) & 3
        r_curve = int(spec['op_kbd_scale_r_curve'][op_spec_idx]) & 3
        packed_data[base + 11] = (r_curve << 2) | l_curve

        # Byte 12: OSC Detune (0-14 from spec -7 to +7), KBD Rate Scaling (0-7)
        # Spec detune is -7 to +7. Packed is 0 to 14.
        osc_detune_packed = min(max(0, int(spec['detune'][op_spec_idx]) + 7), 14)
        kbd_rate_sc = int(spec['op_kbd_rate_scale'][op_spec_idx]) & 7
        packed_data[base + 12] = (osc_detune_packed << 3) | kbd_rate_sc

        # Byte 13: Key Velocity Sens (0-7), Amp Mod Sens (0-3)
        kvs = int(spec['sensitivity'][op_spec_idx]) & 7
        ams = int(spec['op_amp_mod_sens'][op_spec_idx]) & 3
        packed_data[base + 13] = (kvs << 2) | ams # SYSEX F: KVS | AMS

        # Byte 14: Operator Output Level (0-99)
        packed_data[base + 14] = int(spec['ol'][op_spec_idx]) & 0x7F

        # Byte 15: Freq Coarse (0-31), OSC Mode (0-1)
        f_coarse = int(spec['coarse'][op_spec_idx]) & 0x1F
        osc_mode = int(spec['fixed_freq'][op_spec_idx]) & 1
        packed_data[base + 15] = (f_coarse << 1) | osc_mode

        # Byte 16: Freq Fine (0-99)
        packed_data[base + 16] = int(spec['fine'][op_spec_idx]) & 0x7F

    # Pitch EG (Bytes 102-109 in final 128-byte array)
    # PR1-PR4
    packed_data[102] = int(spec['pitch_eg_rate'][0]) & 0x7F
    packed_data[103] = int(spec['pitch_eg_rate'][1]) & 0x7F
    packed_data[104] = int(spec['pitch_eg_rate'][2]) & 0x7F
    packed_data[105] = int(spec['pitch_eg_rate'][3]) & 0x7F
    # PL1-PL4
    packed_data[106] = int(spec['pitch_eg_level'][0]) & 0x7F
    packed_data[107] = int(spec['pitch_eg_level'][1]) & 0x7F
    packed_data[108] = int(spec['pitch_eg_level'][2]) & 0x7F
    packed_data[109] = int(spec['pitch_eg_level'][3]) & 0x7F

    # Algorithm (Byte 110), 0-31
    packed_data[110] = int(spec['algorithm']) & 0x1F

    # OSC Key Sync (0-1), Feedback (0-7) (Byte 111)
    oks = int(spec['osc_key_sync']) & 1
    fb = int(spec['feedback']) & 7
    packed_data[111] = (oks << 3) | fb

    # LFO Parameters
    # Speed (Byte 112)
    packed_data[112] = int(spec['lfo_speed']) & 0x7F
    # Delay (Byte 113)
    packed_data[113] = int(spec['lfo_delay']) & 0x7F
    # Pitch Mod Depth (Byte 114)
    packed_data[114] = int(spec['lfo_pitch_mod_depth']) & 0x7F
    # Amp Mod Depth (Byte 115)
    packed_data[115] = int(spec['lfo_amp_mod_depth']) & 0x7F

    # LFO PMS (0-7), Wave (0-5), Key Sync (0-1) (Byte 116)
    lfo_pms = int(spec['lfo_pitch_mod_sens']) & 7
    lfo_wave = int(spec['lfo_waveform']) & 7 # Wave is 0-5, so 3 bits needed.
    lfo_ks = int(spec['lfo_key_sync']) & 1
    packed_data[116] = (lfo_pms << 4) | (lfo_wave << 1) | lfo_ks

    # Transpose (Byte 117)
    # Spec transpose is relative, packed is 0-48 (24 is no transpose)
    packed_transpose = min(max(0, int(spec['transpose']) + 24), 48)
    packed_data[117] = packed_transpose & 0x7F # Range 0-48 fits in 7 bits.

    # Voice Name (Bytes 118-127)
    name_str = spec['name'][:10].ljust(10) # Ensure 10 chars
    name_ascii = name_str.encode('ascii')
    for i in range(10):
        packed_data[118 + i] = name_ascii[i] & 0x7F

    return packed_data


def write_syx_from_specs(specs_list, output_filepath, user_defaults=None):
    """
    Writes a list of voice specifications to a DX7 SYSEX file (32-voice bulk dump).

    Args:
        specs_list (list): A list of 'spec' dictionaries.
        output_filepath (str): Path to write the .syx file.
        user_defaults (dict, optional): A dictionary to override default parameters
                                        for voices not fully specified or for padding.
    """
    # SYSEX header for 32-voice bulk dump
    # F0 43 0n 09 20 00 (n=channel, typically 0 for ch1)
    # Byte count MSB=0x20, LSB=0x00 means 4096 bytes of voice data
    sysex_header = bytes([0xF0, 0x43, 0x00, 0x09, 0x20, 0x00])
    sysex_footer = bytes([0xF7])

    # Prepare global defaults by merging user_defaults into DEFAULT_SPEC_PARAMS
    global_defaults = copy.deepcopy(DEFAULT_SPEC_PARAMS)
    specs_list = copy.deepcopy(specs_list)
    if user_defaults:
        # A more robust merge might be needed if user_defaults contains nested dicts/lists
        for key, value in user_defaults.items():
            if isinstance(value, dict) and isinstance(global_defaults.get(key), dict):
                 global_defaults[key].update(value)
            elif isinstance(value, list) and isinstance(global_defaults.get(key), list) and key in ['eg_rate', 'eg_level']:
                # Special handling for list of lists like eg_rate, eg_level
                for i in range(len(value)):
                    if i < len(global_defaults[key]):
                        for j in range(len(value[i])):
                            if j < len(global_defaults[key][i]):
                                global_defaults[key][i][j] = value[i][j]
            elif isinstance(value, list) and isinstance(global_defaults.get(key), list):
                # For 1D lists like 'ol', 'coarse', etc.
                for i in range(len(value)):
                    if i < len(global_defaults[key]):
                        global_defaults[key][i] = value[i]
            else:
                global_defaults[key] = value


    all_voice_data = bytearray()
    num_voices_to_write = 32

    for i in range(num_voices_to_write):
        if i < len(specs_list):
            user_s = specs_list[i]
        else:
            # Pad with default voice if specs_list is too short
            user_s = {} # Use an empty spec to get a full default voice
        for key in user_s:
            if key not in ['modmatrix', 'outmatrix']:
                if isinstance(user_s[key], list):
                    user_s[key] = np.array(user_s[key])[...,::-1].tolist()
        resolved_s = _get_resolved_spec(user_s, global_defaults)
        packed_voice = _pack_voice_to_128_bytes(resolved_s)
        all_voice_data.extend(packed_voice)

    # Calculate checksum
    # Sum of all 4096 data bytes, then 2's complement, masked with 0x7F
    checksum_sum = sum(all_voice_data)
    checksum = (-(checksum_sum % 256)) & 0x7F # Modulo 256 for byte sum, then 2's comp, then mask

    # Assemble full SYSEX message
    sysex_message = sysex_header + all_voice_data + bytes([checksum]) + sysex_footer

    # Write to file
    with open(output_filepath, 'wb') as f:
        f.write(sysex_message)

    print(f"SYSEX data for {num_voices_to_write} voices written to {output_filepath}")



# TODO - 1-2-AX working memory task
#class 1-2-AX(SymbolicSequencer):
"""
https://en.wikipedia.org/wiki/1-2-AX_working_memory_task

"""
# def generate_sequence(possible_n_inner=None, n_outer=None, seed=None):

""" generate_sequence() creates a sequence of symbols and responses for the
  1-2-AX continous performance task as described in Frank et al., 2001
    (doi: 10.3758/CABN.1.2.137).\n

    EXAMPLE USAGE:
    sequence, response, out_sequence = generate_sequence(possible_n_inner=np.arange(1, 5), n_outer=50, seed=123)

    INPUTS
    possible_n_inner = int, maximal number of symbol pairs in inner loops of the sequence, 1:1:n_inner
    n_outer          = int, number of outer loops in the sequence
    seed (optional)  = boolean, for deterministic random number generation (default = None)

    OUTPUTS
    sequence =      numpy array, final sequence to be used
    response =      boolean array, correct responses to items in the sequence
    out_sequence =  nd array, output sequence where each element is a symbol pair, useful for checking statistics
"""
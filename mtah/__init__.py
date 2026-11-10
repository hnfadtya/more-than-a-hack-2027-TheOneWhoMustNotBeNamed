"""Our possession pipeline for More than a Hack 2027.

Layout:
    mtah.common         shared match representation (competition + external data)
    mtah.atd            fast reader for the provided ATD, cached as .npz
    mtah.decode         H/A/D Viterbi decoder with minimum durations
    mtah.scoring        wrapper around the official scorer, for any pair of label arrays
    mtah.external.*     converters for labelled tracking from other games
"""

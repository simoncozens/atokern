from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import sys

def create_csv(inpath, outpath):
    sg = {event_accumulator.COMPRESSED_HISTOGRAMS: 1,
          event_accumulator.IMAGES: 1,
          event_accumulator.AUDIO: 1,
          event_accumulator.SCALARS: 0,
          event_accumulator.HISTOGRAMS: 1}
    ea = event_accumulator.EventAccumulator(inpath, size_guidance=sg)
    ea.Reload()
    scalar_tags = ea.Tags()['scalars']
    df = pd.DataFrame(columns=scalar_tags)
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        scalars = np.array(map(lambda x: x.value, events))
        df.loc[:, tag] = scalars
    df.to_csv(outpath)

if __name__ == '__main__':
    args = sys.argv
    inpath = args[1]
    outpath = args[2]
    create_csv(inpath, outpath)

Atokern
=======

A neural network for kerning fonts.

## Getting started

This is a Python 3 script. (I think it would probably work fine with Python 2 with a few minor modifications.)

Install the module requirements with `pip`:

    pip3 install -r requirements.txt

To determine kern pairs for a font:

    python3 neural-kern.py myfont.otf

Currently this only generates kern pairs for alphanumerics (upper and lower case) and some punctuation. To kern other glyphs, edit the `safe_glyphs` array in `neural-kern.py`. It has not been tested on other glyphs, but it ought to work.

## How it works

See http://typedrawers.com/discussion/2428

## Training

The current best version of the neural network model is provided in `kernmodel.hdf5`. The network has been trained on 250 upright Roman fonts, and is achieving around 92% validation accuracy predicting kern pairs. But I'm aware that is probably not enough. I am sure better accuracy could be achieved with a deeper model and more fonts. I don't know whether italics can be kerned with the current model or if you need to train a new model to do that.

To train or tune the network yourself you also need the kerning dump script from Adobe type tools:
https://github.com/adobe-type-tools/kern-dump

Place a bunch of font files (I figure fairly similar ones should be useful, like grotesque sans) into a directory called `kern-dump`, and dump all their kern pairs:

    for i in kern-dump/*.otf ; python dumpkerning.py  $i

Then run:

    python atokern.py

You can alter some of the network settings, and check the range of files which are being processed, in the `settings.py` file.

## Licence

Copyright 2017 Simon Cozens

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
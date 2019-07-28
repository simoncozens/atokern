kerncritic
==========

A neural network for kerning fonts. This has gone through a number of experimental designs and phases. The latest version shows a number of word images to a neural network - some of these are well-kerned, others are not - and asks the network to distinguish which are wrong and how to fix them.

## Getting started

This is a Python 3 script. (I think it would probably work fine with Python 2 with a few minor modifications.)

Install the module requirements with `pip`:

    pip3 install -r requirements.txt

To report on the kerning status of a font:

    ./kerncritic myfont.otf

By default this will test every uppercase basic Latin (A-Z) against every other uppercase basic Latin. To change the range of pairs checked, use the `--left` and `--right` options. You can use the token `<uc>` as a shortcut for A-Z and the token `<lc>` as a shortcut for a-z. For example: `--right '<uc><lc>0123456789'` will test every uppercase basic Latin against basic Latin letters and numerals.

`kerncritic` reports on pairs that it is more than 70% sure are wrong. You can change this with the `--tolerance` option. Use `kerncritic --help` for more.

## Licence

Copyright 2017-2019 Simon Cozens

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
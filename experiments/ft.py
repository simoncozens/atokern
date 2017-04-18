import freetype
import numpy as np
import pickle

face = freetype.Face("/Users/simon/Library/Fonts/SourceSansPro-Regular.otf")
face.set_char_size( 64*256 )

safe_glyphs = set([b"f", b"x",b"y",b"uni01D7"])
# safe_glyphs = set([b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h", b"i", b"j", b"k", b"l", b"m", b"n", b"o", b"p", b"q", b"r", b"s", b"t", b"u", b"v", b"w", b"x", b"y", b"z", b"one", b"two", b"three", b"four", b"five", b"six", b"seven", b"eight", b"nine", b"zero", b"period", b"comma", b"colon"])
loutlines = dict()
routlines = dict()

ascender = int(face.ascender/face.units_per_EM*256)
height = int(face.height/face.units_per_EM*256)
print(height)
def unpack_mono_bitmap(bitmap):
  data = bytearray(bitmap.rows * bitmap.width)
  for y in range(bitmap.rows):
    for byte_index in range(bitmap.pitch):
      byte_value = bitmap.buffer[y * bitmap.pitch + byte_index]
      num_bits_done = byte_index * 8
      rowstart = y * bitmap.width + byte_index * 8
      for bit_index in range(min(8, bitmap.width - num_bits_done)):
        bit = byte_value & (1 << (7 - bit_index))
        data[rowstart + bit_index] = 1 if bit else 0
  return data

def glyph_to_sb(glyph, data, which="L"):
  sb = []
  w, h = glyph.bitmap.width, glyph.bitmap.rows
  if which == "L":
    iterx = range(w)
    last = w-1
    const = glyph.metrics.horiBearingX / 64.0
  else:
    iterx = range(w-1,-1,-1)
    last = 0
    const = glyph.metrics.horiAdvance / 64.0 - (w + glyph.metrics.horiBearingX / 64.0)

  for _ in range(ascender-int(glyph.metrics.horiBearingY / 64)):
    sb.append(int(const+w))

  for y in range(h):
    for x in iterx:
      if data[w*y+x] == 1 or x==last:
        # print("*",end="")
        if which == "L":
          sb.append(int(const+x))
        else:
          sb.append(int(const+(w-x)))
        break
  while len(sb) < height:
    sb.append(0)
  return sb

for g in safe_glyphs:
  glyphindex = face.get_name_index(g)
  if glyphindex:
    print(g)
    face.load_glyph(glyphindex, freetype.FT_LOAD_RENDER |
                              freetype.FT_LOAD_TARGET_MONO)
    data = unpack_mono_bitmap(face.glyph.bitmap)
    loutlines[g] = glyph_to_sb(face.glyph, data, which="L")
    routlines[g] = glyph_to_sb(face.glyph, data, which="R")


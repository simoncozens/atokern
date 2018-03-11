import numpy as np
import freetype
import pickle
import os.path
import sys

safe_glyphs = set([
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
   "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", 
   "period", "comma", "colon"
   ])

samples = 100

def unpack_mono_bitmap(bitmap):
  data = bytearray(bitmap.rows * bitmap.width)
  buff = bitmap._get_buffer()
  for y in range(bitmap.rows):
    for byte_index in range(bitmap.pitch):
      byte_value = buff[y * bitmap.pitch + byte_index]
      num_bits_done = byte_index * 8
      rowstart = y * bitmap.width + byte_index * 8
      for bit_index in range(min(8, bitmap.width - num_bits_done)):
        bit = byte_value & (1 << (7 - bit_index))
        data[rowstart + bit_index] = 1 if bit else 0
  return data

def bbox(outline):
  start, end = 0, 0
  VERTS = []
  # Iterate over each contour
  for i in range(len(outline.contours)):
      end    = outline.contours[i]
      points = outline.points[start:end+1]
      points.append(points[0])
      tags   = outline.tags[start:end+1]
      tags.append(tags[0])
      segments = [ [points[0],], ]
      for j in range(1, len(points) ):
          segments[-1].append(points[j])
          if tags[j] & (1 << 0) and j < (len(points)-1):
              segments.append( [points[j],] )
      verts = [points[0], ]
      for segment in segments:
          if len(segment) == 2:
              verts.extend(segment[1:])
          elif len(segment) == 3:
              verts.extend(segment[1:])
          else:
              verts.append(segment[1])
              for i in range(1,len(segment)-2):
                  A,B = segment[i], segment[i+1]
                  C = ((A[0]+B[0])/2.0, (A[1]+B[1])/2.0)
                  verts.extend([ C, B ])
              verts.append(segment[-1])
      VERTS.extend(verts)
      start = end+1
  if (len(VERTS)<1):
    return (0,0,0,0)
  VERTS = np.array(VERTS)
  x,y = VERTS[:,0], VERTS[:,1]
  VERTS[:,0], VERTS[:,1] = x, y

  xmin, xmax = x.min() / 64, x.max() /64
  ymin, ymax = y.min() / 64, y.max() / 64
  return (xmin, xmax, ymin,ymax)

# Turn a glyph into a tensor of boundary samples
def glyph_to_sb(face, data, which="L"):
  glyph = face.glyph
  sb = []
  w, h = glyph.bitmap.width, glyph.bitmap.rows
  ascender = face.ascender
  lsb = int(glyph.metrics.horiBearingX / 64)
  rsb = glyph.metrics.horiAdvance / 64 - (w + glyph.metrics.horiBearingX/64)
  # print("Width: ", w)
  # print("Height: ", h)
  # print("LSB: ", lsb)
  # print("RSB: ", rsb)
  # print("Ascender", ascender)
  # print("Bearing Y", glyph.metrics.horiBearingY / 64)
  # print("Bbox", bbox(glyph.outline))
  (xmin, xmax, ymin,ymax) = bbox(glyph.outline)
  if which == "L":
    iterx = range(w)
    last = w-1
    const = lsb
  else:
    iterx = range(w-1,-1,-1)
    last = 0
    const = rsb
  # print("Which", which)
  # print("const", const)
  for _ in range(ascender-int(glyph.metrics.horiBearingY / 64)):
    sb.append(w)

  if (ymin>0):
    ymin = 0

  for y in range(-int(ymin),h):
    for x in iterx:
      y2 = int(ymin)+y
      if data[w*y2+x] == 1 or x==last:
        if which == "L":
          sb.append(int(const+x))
        else:
          sb.append(int(const+(w-x)))
        break

  newsb = []
  i = 0
  for i in range(samples):
    sliceval = int(i*len(sb) / samples)
    newsb.append(sb[sliceval])
  # print(newsb)
  return newsb

def loadglyph(face, g):
  glyphindex = face.get_name_index(g.encode("utf8"))
  if glyphindex:
    face.load_glyph(glyphindex, freetype.FT_LOAD_RENDER |
                              freetype.FT_LOAD_TARGET_MONO)
    data = unpack_mono_bitmap(face.glyph.bitmap)
    print(g)
    return np.array(glyph_to_sb(face, data, which="L")), np.array(glyph_to_sb(face, data, which="R"))

def loadfont(path, kerndump):

  def load_kernpairs(file):
    with open(file) as f:
      for line in f:
        l,r,k = line.split()
        if not l in kernpairs:
          kernpairs[l] = dict()
          allglyphs.add(l)
        kernpairs[l][r] = int(k)
        allglyphs.add(r)

  if os.path.isfile(path+".pickle"):
    obj = pickle.load(open(path+".pickle","rb"))
    loutlines, routlines, kernpairs, mwidth = obj["loutlines"], obj["routlines"], obj["kerndata"], obj["mwidth"]
  else:
    import freetype
    face = freetype.Face(path)
    face.set_char_size( 64 * face.units_per_EM )
    mwidth = get_m_width(path)
    loutlines = dict()
    routlines = dict()
    kernpairs = dict()
    allglyphs = set(safe_glyphs)
    for l in safe_glyphs:
      kernpairs[l]=dict()

    if kerndump:
      load_kernpairs(kerndump)
    for g in allglyphs:
      print(g+ " ", end='',flush=True)
      loutlines[g], routlines[g] = loadglyph(face, g)
    # print("")
    obj = {"loutlines": loutlines, "routlines": routlines, "kerndata": kernpairs, "mwidth": mwidth}
    if kerndump:
      pickle.dump(obj, open(path+".pickle","wb"))

  return loutlines, routlines, kernpairs, mwidth

def get_m_width(path):
  import freetype
  face = freetype.Face(path)
  face.set_char_size( 64 * face.units_per_EM )
  n = face.get_name_index("m")
  face.load_glyph(n, freetype.FT_LOAD_RENDER |
                            freetype.FT_LOAD_TARGET_MONO)
  return face.glyph.metrics.horiAdvance / 64

def get_n_width(path):
  face = freetype.Face(path)
  face.set_char_size( 64 * face.units_per_EM )
  n = face.get_name_index("n")
  face.load_glyph(n, freetype.FT_LOAD_RENDER |
                            freetype.FT_LOAD_TARGET_MONO)
  return face.glyph.metrics.horiAdvance / 64

def add_m_width(path):
  obj = pickle.load(open(path+".pickle","rb"))
  obj["mwidth"] = get_m_width(path)
  print(path, obj["mwidth"])
  pickle.dump(obj, open(path+".pickle","wb"))

if __name__ == '__main__':
  for n in sys.argv[1::]:
    print(n+": ", end="")
    loutlines, routlines, kernpairs,mwidth = loadfont(n, n+".kerndump")
    # print("A:", np.array(routlines["A"]))
    # print("V:", np.array(loutlines["V"]))
    #add_m_width(n)

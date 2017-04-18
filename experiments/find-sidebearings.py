import sys
from PIL import Image
im = Image.open(sys.argv[1])
px = im.load()

def get_sb(im, which="L"):
  w,h = im.size
  data = im.getdata()
  if which == "L":
    iterx = range(w)
    last = w-1
  else:
    iterx = range(w-1,-1,-1)
    last = 0
  sb = []
  for x in range(h):
    for y in iterx:
      if data[w*x+y] != 255 or y==last:
        if which == "L":
          sb.append(y)
        else:
          sb.append(w-y)
        break
  return sb

print(get_sb(im,which='R'))
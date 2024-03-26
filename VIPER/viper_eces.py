import random
random.seed(1)
def readD(fn):
  h = {}
  for line in open(fn):
    line = line.strip()
    x = line.split()
    a,b = x[0].strip(),x[1].strip()
    h[a] = b
  return h

def eces(prob, text):
  words = []
  h = readD("VIPER/selected.neighbors")
  truth = text
  text = text.split()
  for line in text:
    if True:
      word = line
      ww = []
      p = random.random()
      if p>prob:
        words.append(( word, word ))
      else:
        max_try = 10
        while max_try:
          w_idx = random.randint(0, len(word)-1)
          if word[w_idx].isalpha():
            break
          max_try -= 1
        if max_try == 0:
          continue
        for wi, w in enumerate(word): 
          if wi==w_idx:
            d = h.get(w,w) 
          else: d=w
          ww.append((d,w))
        words.append(( "".join([c[0] for c in ww]), "".join([c[-1] for c in ww]) ))
  disturbed = " ".join([w[0] for w in words])
  return disturbed
  

let arr = [];
Precision = [];
Recall = [];

function rel(e) {
  arr[arr.length] = e;
  let p, countrel = 0,
    countnrel = 0,
    r;
  for (i = 0; i < arr.length; i++) {
    if (arr[i] == "relevant") {
      countrel += 1;
    } else {
      countnrel += 1;
    }
  }
  p = (countrel / arr.length)
  Precision.push(p)
  alert("precision @ " + arr.length + "is " + p)
  if (arr.length == 10) {
    for (i = 0; i < arr.length; i++) {
      let countrelr = 0,
        countnrelr = 0;
      for (j = 0; j <= i; j++) {
        if (arr[j] == "relevant") {
          countrelr += 1;
        } else {
          countnrelr += 1;
        }
      }
      r = countrelr / countrel;
      Recall.push(r);
      alert("recall at " + (i + 1) + " is " + r);
    }
  }
  return arr;
}


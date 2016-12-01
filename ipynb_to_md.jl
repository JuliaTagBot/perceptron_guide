ipynb_files = readdir(".") |>
  filelist -> filter(filename -> endswith(filename, "ipynb"), filelist) #|>
  # filelist -> map(fname -> "Reports/" * fname, filelist)
# map(fname -> run(`jupyter nbconvert --to markdown Reports/$fname`), ipynb_files)

println(ipynb_files)

map(fname -> run(`/home/guillaume/anaconda3/bin/jupyter nbconvert --to markdown $fname`), ipynb_files)

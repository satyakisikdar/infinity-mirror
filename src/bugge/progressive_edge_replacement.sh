#!/bin/bash

array=( $@ )
len=${#array[@]}
file=${array[$len-1]}
other_args=${array[@]:0:$len-1}

r=0.0
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.0025
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.005
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.01
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.02
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.04
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.08
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.16
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.32
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=0.64
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

r=1.0
end="_r=$r"

{ time python test.py $other_args -r $r; } &> "output_files/$file$end"

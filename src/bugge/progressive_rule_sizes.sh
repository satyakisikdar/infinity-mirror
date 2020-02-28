#!/bin/bash

k_min=2
k_max=2

array=( $@ )
len=${#array[@]}
file=${array[$len-1]}
other_args=${array[@]:0:$len-1}

end="_$k_min$k_max"

{ time python test.py $k_min $k_max $other_args; } &> "output_files/$file$end"

k_max=$((k_max+1))
end="_$k_min$k_max"

{ time python test.py $k_min $k_max $other_args; } &> "output_files/$file$end"

k_max=$((k_max+1))
end="_$k_min$k_max"

{ time python test.py $k_min $k_max $other_args; } &> "output_files/$file$end"

k_max=$((k_max+1))
end="_$k_min$k_max"

{ time python test.py $k_min $k_max $other_args; } &> "output_files/$file$end"

k_max=$((k_max+1))
end="_$k_min$k_max"

{ time python test.py $k_min $k_max $other_args; } &> "output_files/$file$end"

k_max=$((k_max+1))
end="_$k_min$k_max"

{ time python test.py $k_min $k_max $other_args; } &> "output_files/$file$end"

k_max=$((k_max+1))
end="_$k_min$k_max"

{ time python test.py $k_min $k_max $other_args; } &> "output_files/$file$end"

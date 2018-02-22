import glob
import os
import subprocess
import sys


def check_result_sum(first_result_sum, other_result_sum):
    return first_result_sum == other_result_sum

table_sizes = [128, 1024, 10240, 102400, 1024000, 2048000, 4096000, 8192000, 16384000, 32768000, 65536000, 131072000]
selectivities = [1.0, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]

output_file = open("./benchmark.txt", "w")
output_file.write("Algorithm,TableSize,DistinctValues,Runtime,USELESS_HWPF,PAPI_BR_MSP,PAPI_PRF_DM,PAPI_RES_STL,QualifyingRows,ResultSum,ExpectedSelectivity\n")

for table_size in table_sizes:
    should_break = False
    for selectivity in selectivities:
        result_rows = int(selectivity * table_size)
        if result_rows < 1:
            distinct_values = table_size + 1
        else:
            distinct_values = table_size / result_rows
        for i, file in enumerate(glob.glob("./*.out")):
            output = subprocess.check_output([file, str(table_size), str(distinct_values)])
            output = str(output, 'utf-8').rstrip()
            print(output)
            if i == 0:
                first_result_sum = int(output.split(",")[-1])
            if not check_result_sum(first_result_sum, int(output.split(",")[-1])):
                print("FAIL! RESULT SUM MISMATCH!")
                print("%d != %d" % (first_result_sum, int(output.split(",")[-1])))
                sys.exit()

            output_file.write(output)
            output_file.write(",%f" % (selectivity))
            output_file.write("\n")

output_file.close()

all:
	g++-7 -O3 -Wall -Wextra -mno-avx512f -mno-avx512pf -mno-avx512er -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512ifma -mno-avx512vbmi -mavx512vl -mavx -DAVX512=1 -DREG=128 avx_scan.cpp -o ./avxScan_128_avx512.out -lpapi
	g++-7 -O3 -Wall -Wextra -mno-avx512f -mno-avx512pf -mno-avx512er -mno-avx512cd -mno-avx512bw -mno-avx512dq -mno-avx512ifma -mno-avx512vbmi -mavx512vl -mavx2 -DAVX512=1 -DREG=256 avx_scan.cpp -o ./avxScan_256.out -lpapi
	g++-7 -O3 -Wall -Wextra -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -march=skylake-avx512 -DAVX512=1 -DREG=512 avx_scan.cpp -o ./avxScan_512.out -lpapi
	g++-7 -O3 -Wall -Wextra -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -DAUTO_VECTORIZATION=1 sisd_scan.cpp -o ./sisd_auto_vec.out -lpapi
	g++-7 -O3 -Wall -Wextra -mno-mmx -mno-sse2 -mno-sse3 -mno-ssse3 -mno-sse4.1 -mno-sse4.2 -mno-sse4 -mno-avx -mno-avx2 -mno-avx512f -mno-avx512pf -mno-avx512er -mno-avx512cd sisd_scan.cpp -o ./sisd.out -lpapi

clean:
	rm ./*.out
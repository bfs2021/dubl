#!/bin/zsh
# Последовательная серия тяжёлых замеров разделения. Запуск: tests/e2e/run_sep_series.sh
cd "$(dirname "$0")/../.." || exit 1
R=tests/e2e/rb.sh
$R sep_best_A_c51 separate best A c51
$R sm_best_A_c51 sep_metrics A c51 best
for clip in B C; do
  $R sep_fast_${clip}_stereo separate fast $clip stereo
  $R sm_fast_${clip}_stereo sep_metrics $clip stereo fast
  $R sep_best_${clip}_stereo separate best $clip stereo
  $R sm_best_${clip}_stereo sep_metrics $clip stereo best
done
echo SERIES_DONE

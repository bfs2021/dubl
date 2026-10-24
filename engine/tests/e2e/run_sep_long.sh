#!/bin/zsh
# Разделение на 120 с (масштабирование времени и памяти): fast, best.
cd "$(dirname "$0")/../.." || exit 1
R=tests/e2e/rb.sh
$R sep_fast_D_stereo separate fast D stereo
$R sep_best_D_stereo separate best D stereo
echo SERIES3_DONE

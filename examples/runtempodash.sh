#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=21G
# BATCH --mail-user=<user>
#SBATCH --mail-type=ALL

# Submit again 3days from now
sbatch --begin=now+72hour ./submit.sh

PYEXE=$(which python)
# PYEXE=/usr/local/bin/python

# archive TEMPO and reference datasets from NO2 and HCHO in parallel.
# Parallel extraction most beneficial for processing multiple months in
# parallel. Because specejs and months are stored separately, they can be
# independently run. To run a month separately, add START_DATE and END_DATE to call.
SPCS="no2 hcho"
for SPC in ${SPCS}
do
  ${PYEXE} -m tempodash --verbose --spc=${SPC} &> logs/log.${SLURM_JOB_ID}.${SPC} &
done
wait

# Run CSV creater with 20 jobs
${PYEXE}  -m tempodash.aggcsv --jobs=20
err=$?
if [[ $err != 0 ]]; then
    echo Failed at aggcsv
    exit $err
fi

# Run plot creater with 20 jobs
${PYEXE} -m tempodash.aggplot --jobs=20
err=$?
if [[ $err != 0 ]]; then
    echo Failed at aggplot
    exit $err
fi

# Update markdown reports
${PYEXE} -m tempodash.markdown
err=$?
if [[ $err != 0 ]]; then
    echo Failed at markdown
    exit $err
fi
echo "Success: completed all steps without err"

# Markdown reports can be viewed using a previewer as HTML without any processing.
# To convert them to HTML or pdf using pandoc
# but that is not part of the example script
# 
# Below are some pointers:
# pandoc --toc --toc-depth=2 -V geometry:margin=0.5in --embed-resources --standalone -o html/pandora_no2_total_vs_tempo_no2_sum_summary.html markdown/pandora_no2_total_vs_tempo_no2_sum_summary.md
# echo -e "\usepackage{sectsty}\sectionfont{\clearpage}" > titlesec.tex
# pandoc --include-in-header titlesec.tex --toc --toc-depth=2 -V toc-title:"Site Table" -V geometry:margin=0.5in -o pdf/pandora_no2_total_vs_tempo_no2_sum_summary.pdf markdown/pandora_no2_total_vs_tempo_no2_sum_summary.md

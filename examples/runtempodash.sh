#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem-per-cpu=21G
# BATCH --mail-user=<user>
#SBATCH --mail-type=ALL

# Submit again 3days from now
# sbatch --begin=now+72hour ./runtempodash.sh

PYEXE=$(which python)
PYEXE=$(which python3.9)

# archive TEMPO and reference datasets from NO2 and HCHO in parallel.
# Parallel extraction most beneficial for processing multiple months in
# parallel. Because specejs and months are stored separately, they can be
# independently run. To run a month separately, add START_DATE and END_DATE to call.
SPCS="no2 hcho"
for SPC in ${SPCS}
do
  echo Get ${SPC}
  # Normally run all days
  # ${PYEXE} -m tempodash --verbose --spc=${SPC} &> logs/get_${SPC}.log

  # For backfill, run months in parallel
  # for SM in 2023-08-01 2023-09-01 ... 2024-04-01
  # do
  #   EM=$(date -ud "${SM} +1month -1seconds" +"%Y-%m-%dT%H:%M:%S")
  #   ${PYEXE} -m tempodash --verbose --spc=${SPC} ${SM} ${EM} &> logs/get_${SPC}_${SM}.log
  # done
  # wait

  # For testing, run representative days neeed for plotting
  # - weekday in sep
  ${PYEXE} -m tempodash --verbose --spc=${SPC} 2023-09-01T00:00:00 2023-09-01T23:59:59 &> logs/get_${SPC}.log
  # - weekend in dec
  ${PYEXE} -m tempodash --verbose --spc=${SPC} 2023-12-02T00:00:00 2023-12-02T23:59:59 &>> logs/get_${SPC}.log
  # - weekday in mar
  ${PYEXE} -m tempodash --verbose --spc=${SPC} 2024-03-05T00:00:00 2024-03-05T23:59:59 &>> logs/get_${SPC}.log
  wait
done
err=$?
if [[ $err != 0 ]]; then
    echo Failed at get
    exit $err
fi
# Run CSV creater with 20 jobs
echo Summarize ${SPC} in CSV
${PYEXE}  -m tempodash.aggcsv --jobs=20 &> logs/aggcsv.log
err=$?
if [[ $err != 0 ]]; then
    echo Failed at aggcsv
    exit $err
fi

# Run plot creater with 20 jobs
echo Plot ${SPC} in PNG
${PYEXE} -m tempodash.aggplot --jobs=20 &> logs/aggplot.log
err=$?
if [[ $err != 0 ]]; then
    echo Failed at aggplot
    exit $err
fi

# Update markdown reports
${PYEXE} -m tempodash.markdown &> logs/markdown.log
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

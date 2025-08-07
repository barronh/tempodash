TEMPO Validation Dashboard
==========================

These results are from the [tempodash project](https://github.com/barronh/tempodash)
tempodash creates a local archive of TEMPO, TropOMI, Pandora, and AirNow
data for the purposes of evaluating the TEMPO mission. It is used to generate
the reports linked below. The most recent versions are posted periodically.

These reports are a work in progress. Please use the
[issues tracker](https://github.com/barronh/tempodash/issues) to provide feedback.

Report Overview
---------------

Reports show TEMPO statistical performance against a reference dataset for
an L2 variable. Statistical performance is summarized by mean bias (MB), mean
absolute error (MAB), root mean square error (RMSE), and index of agreement
(IOA). In addition, there are boxplots illustrating population distributions.
Each report is a self contained HTML (ie, figures are embedded), which makes
them portable but very large. Understanding the datasets and report types
is useful to navigate efficiently.

### Reference Datasets

The best reference is Pandora, but comparing to TropOMI is a useful benchmark.
The L2 variables include tropospheric column nitrogen dioxide (tropNO2), total
column nitrogen dioxide (totNO2=tropNO2+stratNO2), and total column
formaldehyde (totHCHO).

### Summary vs Details

Summary reports show system-wide performance, while detailed reports also show
performance at individual locations. Both kinds of reports show performance as
a function of month, week, solar zenith angle, cloud fraction, location, and by
variable magnitude. Because of the large number of possible combinations, the
detail reports are 300+ to 600+ pages long, while the summaries are 10 to 20
pages.

Report Links
------------

### Summaries

* [TEMPO vs Pandora totNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_no2_total_vs_tempo_no2_sum_summary.html)
* [TEMPO vs TropOMI totNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/tropomi_offl_no2_sum_vs_tempo_no2_sum_summary.html)
* [TEMPO vs TropOMI tropNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/tropomi_offl_no2_trop_vs_tempo_no2_trop_summary.html)
* [TEMPO vs Pandora totHCHO](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_hcho_total_vs_tempo_hcho_total_summary.html)
* [TEMPO vs TropOMI totHCHO](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/tropomi_offl_hcho_total_vs_tempo_hcho_total_summary.html)
* [TropOMI vs Pandora totNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_no2_total_vs_tropomi_offl_no2_sum_summary.html)
* [TropOMI vs Pandora totHCHO](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_hcho_total_vs_tropomi_offl_hcho_total_summary.html)

### Details

* [TEMPO vs Pandora totNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_no2_total_vs_tempo_no2_sum.html)
* [TEMPO vs TropOMI totNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/tropomi_offl_no2_sum_vs_tempo_no2_sum.html)
* [TEMPO vs TropOMI tropNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/tropomi_offl_no2_trop_vs_tempo_no2_trop.html)
* [TEMPO vs Pandora totHCHO](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_hcho_total_vs_tempo_hcho_total.html)
* [TEMPO vs TropOMI totHCHO](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/tropomi_offl_hcho_total_vs_tempo_hcho_total.html)
* [TropOMI vs Pandora totNO2](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_no2_total_vs_tropomi_offl_no2_sum.html)
* [TropOMI vs Pandora totHCHO](https://gaftp.epa.gov/Air/aqmg/bhenders/share/TEMPO/pandora_hcho_total_vs_tropomi_offl_hcho_total.html)

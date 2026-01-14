## Numerical Features
|                      |   count |          mean |           std |          min |         25% |         50% |         75% |             max |
|:---------------------|--------:|--------------:|--------------:|-------------:|------------:|------------:|------------:|----------------:|
| ID                   |  148670 |  99224.5      |  42917.5      | 24890        |  62057.2    |  99224.5    | 136392      | 173559          |
| year                 |  148670 |   2019        |      0        |  2019        |   2019      |   2019      |   2019      |   2019          |
| loan_amount          |  148670 | 331118        | 183909        | 16500        | 196500      | 296500      | 436500      |      3.5765e+06 |
| rate_of_interest     |  112231 |      4.04548  |      0.561391 |     0        |      3.625  |      3.99   |      4.375  |      8          |
| Interest_rate_spread |  112031 |      0.441656 |      0.513043 |    -3.638    |      0.076  |      0.3904 |      0.7754 |      3.357      |
| Upfront_charges      |  109028 |   3225        |   3251.12     |     0        |    581.49   |   2596.45   |   4812.5    |  60000          |
| term                 |  148629 |    335.137    |     58.4091   |    96        |    360      |    360      |    360      |    360          |
| property_value       |  133572 | 497893        | 359935        |  8000        | 268000      | 418000      | 628000      |      1.6508e+07 |
| income               |  139520 |   6957.34     |   6496.59     |     0        |   3720      |   5760      |   8520      | 578580          |
| Credit_Score         |  148670 |    699.789    |    115.876    |   500        |    599      |    699      |    800      |    900          |
| LTV                  |  133572 |     72.7465   |     39.9676   |     0.967478 |     60.4749 |     75.1359 |     86.1842 |   7831.25       |
| Status               |  148670 |      0.246445 |      0.430942 |     0        |      0      |      0      |      0      |      1          |
| dtir1                |  124549 |     37.7329   |     10.5454   |     5        |     31      |     39      |     45      |     61          |

## Categorical Features
|                           |   count |   unique |   missing | most_frequent   |   most_frequent_freq |
|:--------------------------|--------:|---------:|----------:|:----------------|---------------------:|
| loan_limit                |  145326 |        2 |      3344 | cf              |               135348 |
| Gender                    |  148670 |        4 |         0 | Male            |                42346 |
| approv_in_adv             |  147762 |        2 |       908 | nopre           |               124621 |
| loan_type                 |  148670 |        3 |         0 | type1           |               113173 |
| loan_purpose              |  148536 |        4 |       134 | p3              |                55934 |
| Credit_Worthiness         |  148670 |        2 |         0 | l1              |               142344 |
| open_credit               |  148670 |        2 |         0 | nopc            |               148114 |
| business_or_commercial    |  148670 |        2 |         0 | nob/c           |               127908 |
| Neg_ammortization         |  148549 |        2 |       121 | not_neg         |               133420 |
| interest_only             |  148670 |        2 |         0 | not_int         |               141560 |
| lump_sum_payment          |  148670 |        2 |         0 | not_lpsm        |               145286 |
| construction_type         |  148670 |        2 |         0 | sb              |               148637 |
| occupancy_type            |  148670 |        3 |         0 | pr              |               138201 |
| Secured_by                |  148670 |        2 |         0 | home            |               148637 |
| total_units               |  148670 |        4 |         0 | 1U              |               146480 |
| credit_type               |  148670 |        4 |         0 | CIB             |                48152 |
| co-applicant_credit_type  |  148670 |        2 |         0 | CIB             |                74392 |
| age                       |  148470 |        7 |       200 | 45-54           |                34720 |
| submission_of_application |  148470 |        2 |       200 | to_inst         |                95814 |
| Region                    |  148670 |        4 |         0 | North           |                74722 |
| Security_Type             |  148670 |        2 |         0 | direct          |               148637 |

## Missing numerical values Report
|                      |         0 |
|:---------------------|----------:|
| Upfront_charges      | 26.6698   |
| Interest_rate_spread | 24.6443   |
| rate_of_interest     | 24.5132   |
| dtir1                | 16.1885   |
| property_value       | 10.0903   |
| LTV                  | 10.0903   |
| income               |  6.184    |
| term                 |  0.027746 |
| loan_amount          |  0        |
| Credit_Score         |  0        |

## Missing categorical values Report
|                           |         0 |
|:--------------------------|----------:|
| loan_limit                | 2.26677   |
| approv_in_adv             | 0.601164  |
| submission_of_application | 0.131163  |
| age                       | 0.131163  |
| loan_purpose              | 0.0866012 |
| Neg_ammortization         | 0.0807157 |
| occupancy_type            | 0         |
| Region                    | 0         |
| co-applicant_credit_type  | 0         |
| credit_type               | 0         |
| total_units               | 0         |
| Secured_by                | 0         |
| lump_sum_payment          | 0         |
| construction_type         | 0         |
| Gender                    | 0         |
| interest_only             | 0         |
| business_or_commercial    | 0         |
| open_credit               | 0         |
| Credit_Worthiness         | 0         |
| loan_type                 | 0         |
| Security_Type             | 0         |

## Skewness report
|                      |            0 |
|:---------------------|-------------:|
| LTV                  | 118.773      |
| income               |  18.1099     |
| property_value       |   4.52721    |
| Upfront_charges      |   1.75189    |
| loan_amount          |   1.56751    |
| rate_of_interest     |   0.385793   |
| Interest_rate_spread |   0.278974   |
| Credit_Score         |   0.00402592 |
| dtir1                |  -0.551263   |
| term                 |  -2.17124    |

## Correlation matrix
|                      |   loan_amount |   rate_of_interest |   Interest_rate_spread |   Upfront_charges |         term |   property_value |      income |   Credit_Score |          LTV |       dtir1 |
|:---------------------|--------------:|-------------------:|-----------------------:|------------------:|-------------:|-----------------:|------------:|---------------:|-------------:|------------:|
| loan_amount          |    1          |       -0.151069    |            -0.378722   |        0.0655075  |  0.175372    |       0.732269   |  0.446324   |    0.00580836  |  0.0372985   |  0.0133088  |
| rate_of_interest     |   -0.151069   |        1           |             0.615815   |       -0.0766747  |  0.209868    |      -0.122024   | -0.0406647  |   -0.000687989 | -0.000372295 |  0.0581444  |
| Interest_rate_spread |   -0.378722   |        0.615815    |             1          |        0.0337286  | -0.15646     |      -0.335879   | -0.149874   |   -0.00256948  |  0.0378228   |  0.0797268  |
| Upfront_charges      |    0.0655075  |       -0.0766747   |             0.0337286  |        1          | -0.0548877   |       0.0525824  |  0.0161168  |   -0.00137411  | -0.0303908   |  0.00188802 |
| term                 |    0.175372   |        0.209868    |            -0.15646    |       -0.0548877  |  1           |       0.0465522  | -0.0520661  |   -0.000873938 |  0.0995706   |  0.111259   |
| property_value       |    0.732269   |       -0.122024    |            -0.335879   |        0.0525824  |  0.0465522   |       1          |  0.406663   |    0.00303425  | -0.201587    | -0.0579288  |
| income               |    0.446324   |       -0.0406647   |            -0.149874   |        0.0161168  | -0.0520661   |       0.406663   |  1          |    0.00326753  | -0.0675254   | -0.267653   |
| Credit_Score         |    0.00580836 |       -0.000687989 |            -0.00256948 |       -0.00137411 | -0.000873938 |       0.00303425 |  0.00326753 |    1           | -0.00614269  | -0.00130483 |
| LTV                  |    0.0372985  |       -0.000372295 |             0.0378228  |       -0.0303908  |  0.0995706   |      -0.201587   | -0.0675254  |   -0.00614269  |  1           |  0.156447   |
| dtir1                |    0.0133088  |        0.0581444   |             0.0797268  |        0.00188802 |  0.111259    |      -0.0579288  | -0.267653   |   -0.00130483  |  0.156447    |  1          |

## Categorical features unique values
|                           |   0 |
|:--------------------------|----:|
| age                       |   7 |
| Region                    |   4 |
| loan_purpose              |   4 |
| credit_type               |   4 |
| total_units               |   4 |
| Gender                    |   4 |
| occupancy_type            |   3 |
| loan_type                 |   3 |
| loan_limit                |   2 |
| submission_of_application |   2 |
| co-applicant_credit_type  |   2 |
| Secured_by                |   2 |
| lump_sum_payment          |   2 |
| construction_type         |   2 |
| interest_only             |   2 |
| Neg_ammortization         |   2 |
| business_or_commercial    |   2 |
| open_credit               |   2 |
| Credit_Worthiness         |   2 |
| approv_in_adv             |   2 |
| Security_Type             |   2 |

## Category distribution report
| feature                   | category          |   percentage |
|:--------------------------|:------------------|-------------:|
| loan_limit                | cf                |   93.1177    |
| loan_limit                | ncf               |    6.88231   |
| Gender                    | Male              |   28.4052    |
| Gender                    | Joint             |   27.9268    |
| Gender                    | Sex Not Available |   25.2363    |
| Gender                    | Female            |   18.4318    |
| approv_in_adv             | nopre             |   84.271     |
| approv_in_adv             | pre               |   15.729     |
| loan_type                 | type1             |   76.1393    |
| loan_type                 | type2             |   13.9251    |
| loan_type                 | type3             |    9.9356    |
| loan_purpose              | p3                |   37.6082    |
| loan_purpose              | p4                |   36.9148    |
| loan_purpose              | p1                |   23.2511    |
| loan_purpose              | p2                |    2.22581   |
| Credit_Worthiness         | l1                |   95.7591    |
| Credit_Worthiness         | l2                |    4.24094   |
| open_credit               | nopc              |   99.625     |
| open_credit               | opc               |    0.374992  |
| business_or_commercial    | nob/c             |   86.0749    |
| business_or_commercial    | b/c               |   13.9251    |
| Neg_ammortization         | not_neg           |   89.7745    |
| Neg_ammortization         | neg_amm           |   10.2255    |
| interest_only             | not_int           |   95.2495    |
| interest_only             | int_only          |    4.75045   |
| lump_sum_payment          | not_lpsm          |   97.7669    |
| lump_sum_payment          | lpsm              |    2.23313   |
| construction_type         | sb                |   99.9807    |
| construction_type         | mh                |    0.0193381 |
| occupancy_type            | pr                |   92.9307    |
| occupancy_type            | ir                |    4.93795   |
| occupancy_type            | sr                |    2.1314    |
| Secured_by                | home              |   99.9807    |
| Secured_by                | land              |    0.0193381 |
| total_units               | 1U                |   98.5194    |
| total_units               | 2U                |    1.00222   |
| total_units               | 3U                |    0.258963  |
| total_units               | 4U                |    0.219446  |
| credit_type               | CIB               |   32.4267    |
| credit_type               | CRIF              |   29.5209    |
| credit_type               | EXP               |   27.8301    |
| credit_type               | EQUI              |   10.2223    |
| co-applicant_credit_type  | CIB               |   50.127     |
| co-applicant_credit_type  | EXP               |   49.873     |
| age                       | 45-54             |   23.3583    |
| age                       | 35-44             |   22.0534    |
| age                       | 55-64             |   21.9633    |
| age                       | 65-74             |   13.8988    |
| age                       | 25-34             |   12.9778    |
| age                       | >74               |    4.83667   |
| age                       | <25               |    0.91177   |
| submission_of_application | to_inst           |   64.5319    |
| submission_of_application | not_inst          |   35.4681    |
| Region                    | North             |   50.2556    |
| Region                    | south             |   43.0484    |
| Region                    | central           |    5.84768   |
| Region                    | North-East        |    0.848355  |
| Security_Type             | direct            |   99.9807    |
| Security_Type             | Indriect          |    0.0193381 |
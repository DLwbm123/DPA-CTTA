# R2 experiment report

R2_EXPERIMENT_COMPLETE

Primary endpoint: remaining_dev, OD/OC mean, four domains equally weighted, then four orders equally weighted.

| Arm | Order 0 | Order 1 | Order 2 | Order 3 | Mean | Mean gain C (pp) |
|---|---:|---:|---:|---:|---:|---:|
| C | 79.1043 | 78.0063 | 78.0079 | 79.3261 | 78.6112 | 0.0000 |
| C_PCA_REGION | 79.2901 | 77.8716 | 77.9686 | 79.9895 | 78.7799 | 0.1688 |
| R2_E | 79.3350 | 77.8649 | 77.9614 | 79.9848 | 78.7865 | 0.1754 |
| R2_D | 79.0835 | 77.9808 | 78.0016 | 79.2967 | 78.5906 | -0.0205 |
| R2_DE | 79.0841 | 77.9856 | 77.9964 | 79.2977 | 78.5910 | -0.0202 |
| R2_F | 79.0474 | 78.0016 | 78.0232 | 79.2714 | 78.5859 | -0.0253 |
| R2_DE_S | 79.0523 | 77.9756 | 77.9948 | 79.2731 | 78.5739 | -0.0372 |

## Matched comparisons

```json
{
  "R2_E-C": 0.17536171887261887,
  "R2_E-C_PCA_REGION": 0.006601294369492194,
  "R2_D-C": -0.020512823164747118,
  "R2_D-C_PCA_REGION": -0.1892732476678738,
  "R2_DE-C": -0.020207158005348447,
  "R2_DE-C_PCA_REGION": -0.18896758250847512,
  "R2_F-C": -0.02525625360211947,
  "R2_F-C_PCA_REGION": -0.19401667810524614,
  "R2_DE_S-C": -0.037218714698994404,
  "R2_DE_S-C_PCA_REGION": -0.20597913920212108,
  "R2_DE-R2_D": 0.00030566515939867145,
  "R2_DE-R2_E": -0.1955688768779673,
  "R2_DE-R2_F": 0.005049095596771025,
  "R2_DE-R2_DE_S": 0.017011556693645957
}
```

Factorial interaction (descriptive pp): -0.006295629210093523

## Focus domains

REFUGE_Valid OC and ORIGA / Drishti_GS gains and costs are detailed by order below. ASSD means are conditional on defined values; paired ASSD uses jointly defined contents only.

| Order | Domain | Arm | OC Dice % | OC ASSD px | OC Dice decline vs C | Macro gain C (pp) |
|---|---|---|---:|---:|---:|---:|
| 0 | REFUGE_Valid | R2_E | 58.4584 | 19.273578164718174 | 0.96875 | -0.7807 |
| 0 | REFUGE_Valid | R2_D | 59.6670 | 18.43185017891033 | 0.9279891304347826 | -0.1334 |
| 0 | REFUGE_Valid | R2_DE | 59.6729 | 18.424162687606923 | 0.9157608695652174 | -0.1312 |
| 0 | REFUGE_Valid | R2_F | 59.7861 | 18.269326326678314 | 0.19293478260869565 | -0.0660 |
| 0 | REFUGE_Valid | R2_DE_S | 59.7076 | 18.40011604182682 | 0.6317934782608695 | -0.1094 |
| 0 | ORIGA | R2_E | 72.9990 | 16.588200985152 | 0.17064846416382254 | 0.2464 |
| 0 | ORIGA | R2_D | 72.4870 | 16.86611039747405 | 0.4129692832764505 | 0.0365 |
| 0 | ORIGA | R2_DE | 72.4858 | 16.865389520571775 | 0.40784982935153585 | 0.0359 |
| 0 | ORIGA | R2_F | 72.4655 | 16.868235868670805 | 0.5221843003412969 | 0.0310 |
| 0 | ORIGA | R2_DE_S | 72.4440 | 16.88416471393263 | 0.6518771331058021 | -0.0037 |
| 0 | Drishti_GS | R2_E | 72.0243 | 21.838961214773967 | 0.13513513513513514 | 1.4424 |
| 0 | Drishti_GS | R2_D | 69.2748 | 23.537514969141462 | 0.7297297297297297 | 0.0085 |
| 0 | Drishti_GS | R2_DE | 69.2755 | 23.53512479831794 | 0.7567567567567568 | 0.0088 |
| 0 | Drishti_GS | R2_F | 68.8941 | 23.72220829640185 | 0.8648648648648649 | -0.1949 |
| 0 | Drishti_GS | R2_DE_S | 69.0862 | 23.621145290519085 | 0.8648648648648649 | -0.0964 |
| 1 | REFUGE_Valid | R2_E | 64.0691 | 15.678723657794915 | 0.9279891304347826 | -0.3171 |
| 1 | REFUGE_Valid | R2_D | 64.4908 | 15.379020115436266 | 0.6847826086956522 | -0.0416 |
| 1 | REFUGE_Valid | R2_DE | 64.4912 | 15.380210606307866 | 0.6793478260869565 | -0.0381 |
| 1 | REFUGE_Valid | R2_F | 64.5207 | 15.352642405229885 | 0.3491847826086957 | -0.0140 |
| 1 | REFUGE_Valid | R2_DE_S | 64.4999 | 15.37385440053363 | 0.5149456521739131 | -0.0306 |
| 1 | ORIGA | R2_E | 70.7638 | 18.155812368342545 | 0.2030716723549488 | 0.4428 |
| 1 | ORIGA | R2_D | 69.7829 | 18.60817598110644 | 0.7952218430034129 | -0.0320 |
| 1 | ORIGA | R2_DE | 69.8017 | 18.599171329694844 | 0.726962457337884 | -0.0205 |
| 1 | ORIGA | R2_F | 69.7954 | 18.582158528775704 | 0.5665529010238908 | -0.0048 |
| 1 | ORIGA | R2_DE_S | 69.7320 | 18.631781121429132 | 0.825938566552901 | -0.0647 |
| 1 | Drishti_GS | R2_E | 61.2966 | 27.06029042106215 | 0.05405405405405406 | 0.0053 |
| 1 | Drishti_GS | R2_D | 61.2792 | 27.06125390614102 | 0.3783783783783784 | -0.0000 |
| 1 | Drishti_GS | R2_DE | 61.2791 | 27.060674924313567 | 0.35135135135135137 | -0.0001 |
| 1 | Drishti_GS | R2_F | 61.2783 | 27.064443841853052 | 0.35135135135135137 | 0.0010 |
| 1 | Drishti_GS | R2_DE_S | 61.2805 | 27.062861502871662 | 0.35135135135135137 | 0.0005 |
| 2 | REFUGE_Valid | R2_E | 60.9680 | 17.51628083660487 | 0.9551630434782609 | -0.6270 |
| 2 | REFUGE_Valid | R2_D | 61.9358 | 16.897977905836196 | 0.595108695652174 | -0.0615 |
| 2 | REFUGE_Valid | R2_DE | 61.9137 | 16.915727042715883 | 0.8288043478260869 | -0.0891 |
| 2 | REFUGE_Valid | R2_F | 61.9972 | 16.81633787406184 | 0.19293478260869565 | -0.0416 |
| 2 | REFUGE_Valid | R2_DE_S | 61.9337 | 16.90810273020697 | 0.59375 | -0.0735 |
| 2 | ORIGA | R2_E | 68.7904 | 18.49505580482943 | 0.1552901023890785 | 0.0964 |
| 2 | ORIGA | R2_D | 68.6249 | 18.582064206904253 | 0.35494880546075086 | 0.0141 |
| 2 | ORIGA | R2_DE | 68.6252 | 18.582275028925224 | 0.35494880546075086 | 0.0144 |
| 2 | ORIGA | R2_F | 68.6558 | 18.56520890067878 | 0.2645051194539249 | 0.0484 |
| 2 | ORIGA | R2_DE_S | 68.6214 | 18.581833872075045 | 0.3856655290102389 | 0.0068 |
| 2 | Drishti_GS | R2_E | 65.2656 | 25.312037396436754 | 0.05405405405405406 | 0.4196 |
| 2 | Drishti_GS | R2_D | 64.3944 | 25.799240765092993 | 0.5945945945945946 | 0.0132 |
| 2 | Drishti_GS | R2_DE | 64.4000 | 25.797485469025162 | 0.5135135135135135 | 0.0185 |
| 2 | Drishti_GS | R2_F | 64.4154 | 25.748686185279354 | 0.4594594594594595 | 0.0428 |
| 2 | Drishti_GS | R2_DE_S | 64.3969 | 25.80222618869539 | 0.5945945945945946 | 0.0088 |
| 3 | REFUGE_Valid | R2_E | 65.0793 | 15.034408305450691 | 0.8777173913043478 | -0.3091 |
| 3 | REFUGE_Valid | R2_D | 65.5638 | 14.705450092022108 | 0.686141304347826 | -0.0355 |
| 3 | REFUGE_Valid | R2_DE | 65.5629 | 14.706019451071404 | 0.7051630434782609 | -0.0351 |
| 3 | REFUGE_Valid | R2_F | 65.5849 | 14.697258631080341 | 0.4673913043478261 | -0.0041 |
| 3 | REFUGE_Valid | R2_DE_S | 65.5783 | 14.698035993827135 | 0.4279891304347826 | -0.0162 |
| 3 | ORIGA | R2_E | 71.3616 | 17.997438805486706 | 0.17235494880546076 | 0.9423 |
| 3 | ORIGA | R2_D | 69.4087 | 18.878265392296438 | 0.7542662116040956 | -0.0443 |
| 3 | ORIGA | R2_DE | 69.4025 | 18.88432447532373 | 0.7679180887372014 | -0.0463 |
| 3 | ORIGA | R2_F | 69.4011 | 18.866040771056007 | 0.5409556313993175 | -0.0167 |
| 3 | ORIGA | R2_DE_S | 69.3421 | 18.904657392350664 | 0.7679180887372014 | -0.0780 |
| 3 | Drishti_GS | R2_E | 64.7675 | 25.143683992489716 | 0.05405405405405406 | 2.1475 |
| 3 | Drishti_GS | R2_D | 60.3698 | 27.09613155262685 | 0.8918918918918919 | -0.0470 |
| 3 | Drishti_GS | R2_DE | 60.3841 | 27.078333704221198 | 0.8648648648648649 | -0.0416 |
| 3 | Drishti_GS | R2_F | 60.1811 | 27.229028661927693 | 0.8918918918918919 | -0.1948 |
| 3 | Drishti_GS | R2_DE_S | 60.2496 | 27.128969242844345 | 0.8918918918918919 | -0.1185 |

## Resource-reference assessment

```json
{
  "arms": {
    "R2_E": {
      "mean_gain_C_pp": 0.17536171887261887,
      "positive_orders": 2,
      "practical_signal": false,
      "risk_warning": false,
      "domain_mean_deltas": {
        "REFUGE": -0.2257617808930199,
        "ORIGA": 0.43198598638517893,
        "REFUGE_Valid": -0.5084845705963588,
        "Drishti_GS": 1.0037072405946788
      }
    },
    "R2_D": {
      "mean_gain_C_pp": -0.020512823164747118,
      "positive_orders": 0,
      "practical_signal": false,
      "risk_warning": false,
      "domain_mean_deltas": {
        "REFUGE": -0.00128891915962015,
        "ORIGA": -0.006411780319723448,
        "REFUGE_Valid": -0.06801726909761996,
        "Drishti_GS": -0.006333324082032021
      }
    },
    "R2_DE": {
      "mean_gain_C_pp": -0.020207158005348447,
      "positive_orders": 0,
      "practical_signal": false,
      "risk_warning": false,
      "domain_mean_deltas": {
        "REFUGE": 0.0003160710095855279,
        "ORIGA": -0.004127655327735624,
        "REFUGE_Valid": -0.07340400652670098,
        "Drishti_GS": -0.0036130411765533665
      }
    },
    "R2_F": {
      "mean_gain_C_pp": -0.02525625360211947,
      "positive_orders": 1,
      "practical_signal": false,
      "risk_warning": false,
      "domain_mean_deltas": {
        "REFUGE": 0.0023640749549365125,
        "ORIGA": 0.01447369204191773,
        "REFUGE_Valid": -0.031407977055433633,
        "Drishti_GS": -0.0864548043499127
      }
    },
    "R2_DE_S": {
      "mean_gain_C_pp": -0.037218714698994404,
      "positive_orders": 0,
      "practical_signal": false,
      "risk_warning": false,
      "domain_mean_deltas": {
        "REFUGE": -0.005138037615811442,
        "ORIGA": -0.0349143465002868,
        "REFUGE_Valid": -0.0574181107542131,
        "Drishti_GS": -0.05140436392566272
      }
    }
  },
  "DE_matched_necessity_supported": false,
  "status": "DESCRIPTIVE_ONLY",
  "next_execution_authorized": false,
  "note": "Shared exposed contents; thresholds are resource references, not significance or execution gates."
}
```

## Counts and limits

```json
{
  "formal": {
    "new_records": 39020,
    "forwards": 312160,
    "backwards": 39020,
    "adam": 39020
  },
  "smoke": {
    "forwards": 224,
    "backwards": 28,
    "base_adam": 28,
    "perturb": 0,
    "restore": 0
  },
  "historical_primary": 15608
}
```

- Exposed development data; four orders share contents.
- Scalar replay checks W/Q/counts/refresh, not feature covariance or ASSD geometry.
- F includes shadow PCA cost; no direction in its loss.
- Same rule does not ensure identical F/DE tokens after adaptation.
- No automatic next run.
- Drishti_GS remaining_dev has 37 contents and one-quarter domain weight; four orders are not independent samples.
- User-authorized IO continuation; prior failures preserved. Formal table excludes interrupted prefixes. Multiple runtime commits; repeated prefixes and fresh smoke exceed the original compute budget explicitly. Terminated workers may each have one unrecorded in-flight visit; physical totals are bounded, not exact.

Full per-subset, domain, channel, paired-tail and mechanism scalars are in public_aggregate.json. No automatic next experiment.

## IO continuation accounting

```json
{
  "source_binding": {
    "run_id": "8ce180e3db2249c19bba9d103ede9cb2",
    "code_sha": "0d515328a6cc42d8e0c6a458265b41e41c454fa6",
    "science_sha256": "882323714fc29b439bccb540cfe7e685733da1e7f0012af8a202954ec2e12353",
    "registration_digest": "8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf"
  },
  "carried_jobs": [
    "o0a0",
    "o0a1",
    "o0a2",
    "o0a3",
    "o0a4",
    "o1a0",
    "o1a1",
    "o1a2",
    "o1a3",
    "o1a4",
    "o2a0",
    "o2a1",
    "o2a2",
    "o2a3",
    "o3a0",
    "o2a4"
  ],
  "executed_jobs": [
    "o3a1",
    "o3a2",
    "o3a3",
    "o3a4"
  ],
  "discarded_prefix_records": 4085,
  "discarded_prefix_physical": {
    "forwards": 32680,
    "backwards": 4085,
    "base_adam": 4085,
    "perturb": 0,
    "restore": 0
  },
  "prior_smoke_physical": {
    "forwards": 672,
    "backwards": 84,
    "base_adam": 84,
    "perturb": 0,
    "restore": 0
  },
  "note": "User-authorized IO continuation; prior failures preserved. Formal table excludes interrupted prefixes. Multiple runtime commits; repeated prefixes and fresh smoke exceed the original compute budget explicitly. Terminated workers may each have one unrecorded in-flight visit; physical totals are bounded, not exact.",
  "abandoned_binding": {
    "run_id": "a8693fe3eab44ddb802c6de6c0caafda",
    "code_sha": "66eea7e880e16d4d25efa4edabc9d8ad59ff175d",
    "science_sha256": "882323714fc29b439bccb540cfe7e685733da1e7f0012af8a202954ec2e12353",
    "registration_digest": "8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf"
  },
  "abandoned_prefixes": {
    "o2a4": 1048,
    "o3a2": 1053
  },
  "actual_total_physical_lower_bound": {
    "forwards": 345736,
    "backwards": 43217,
    "base_adam": 43217
  },
  "actual_total_physical_upper_bound": {
    "forwards": 345752,
    "backwards": 43219,
    "base_adam": 43219
  },
  "unrecorded_inflight_upper_bound": {
    "forwards": 16,
    "backwards": 2,
    "base_adam": 2
  },
  "recorded_scoring_visits": 43105,
  "partial_binding": {
    "run_id": "6bb3ccee255243a684c95975ba2671bc",
    "code_sha": "1560f28d41dda5665404ececbefc4f0a7c69d192",
    "science_sha256": "882323714fc29b439bccb540cfe7e685733da1e7f0012af8a202954ec2e12353",
    "registration_digest": "8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf"
  },
  "partial_prefixes": {
    "o3a2": 1408
  },
  "additional_carried_jobs": [
    "o2a4"
  ]
}
```

# R1 results

R1_EXPERIMENT_COMPLETE

{
  "physical": {
    "records": 46824,
    "forwards": 398004,
    "backwards": 46824,
    "adam": 46824
  },
  "descriptive_matched_comparisons": {
    "C_SENS-C": 0.0,
    "C_PER256-C": -1.5498864059009172,
    "C_SENS-C_PER256": 1.5498864059009172,
    "C_PCA_GLOBAL-C": -0.09465109779889858,
    "C_PCA_REGION-C": 0.16876042450312312,
    "C_PCA_REGION-C_PCA_GLOBAL": 0.2634115223020217,
    "C_PCA_REGION-C_PCA_SHUFFLED": 0.2649877199309074
  },
  "assessment_including_risk_and_inactive": {
    "status": "MIXED",
    "arms": {
      "C_PER256": {
        "mean_gain_pp": -1.5498864059009172,
        "positive_orders": 0,
        "domain_mean_deltas": {
          "REFUGE": -0.9324004454698418,
          "ORIGA": -5.970527335924263,
          "REFUGE_Valid": 3.839936507680182,
          "Drishti_GS": -3.136554349889746
        },
        "worst_order_score_delta_pp": -1.1412575825702902,
        "active": true,
        "practical_signal": false,
        "risk_warning": true
      },
      "C_SENS": {
        "mean_gain_pp": 0.0,
        "positive_orders": 0,
        "domain_mean_deltas": {
          "REFUGE": 0.0,
          "ORIGA": 0.0,
          "REFUGE_Valid": 0.0,
          "Drishti_GS": 0.0
        },
        "worst_order_score_delta_pp": 0.0,
        "active": false,
        "practical_signal": false,
        "risk_warning": false
      },
      "C_PCA_GLOBAL": {
        "mean_gain_pp": -0.09465109779889858,
        "positive_orders": 0,
        "domain_mean_deltas": {
          "REFUGE": -0.0019366412215902074,
          "ORIGA": -0.08127338566004383,
          "REFUGE_Valid": -0.23125710479614625,
          "Drishti_GS": -0.06413725951780691
        },
        "worst_order_score_delta_pp": -0.09394561488366548,
        "active": true,
        "practical_signal": false,
        "risk_warning": false
      },
      "C_PCA_SHUFFLED": {
        "mean_gain_pp": -0.0962272954277843,
        "positive_orders": 0,
        "domain_mean_deltas": {
          "REFUGE": -0.0008892701851017648,
          "ORIGA": -0.08746217771394171,
          "REFUGE_Valid": -0.22720052211713693,
          "Drishti_GS": -0.06935721169493902
        },
        "worst_order_score_delta_pp": -0.09791296845227748,
        "active": true,
        "practical_signal": false,
        "risk_warning": false
      },
      "C_PCA_REGION": {
        "mean_gain_pp": 0.16876042450312312,
        "positive_orders": 2,
        "domain_mean_deltas": {
          "REFUGE": -0.18963518965229653,
          "ORIGA": 0.3897327288276138,
          "REFUGE_Valid": -0.4635051059520805,
          "Drishti_GS": 0.9384492647892628
        },
        "worst_order_score_delta_pp": -0.1347894423383309,
        "active": true,
        "practical_signal": false,
        "risk_warning": false
      }
    },
    "candidate_routes": [],
    "recommendation": "Retain C; no sufficient matched evidence",
    "combination_run_authorized": false,
    "note": "Frozen descriptive resource-allocation thresholds, not significance, clinical criteria, or execution gates. INACTIVE mechanisms are identified by active=false; no retuning."
  },
  "secondary_controls": {
    "0": {
      "A": {
        "status": "AVAILABLE"
      },
      "C0": {
        "status": "AVAILABLE"
      }
    },
    "1": {
      "A": {
        "status": "AVAILABLE"
      },
      "C0": {
        "status": "AVAILABLE"
      }
    },
    "2": {
      "A": {
        "status": "AVAILABLE"
      },
      "C0": {
        "status": "AVAILABLE"
      }
    },
    "3": {
      "A": {
        "status": "AVAILABLE"
      },
      "C0": {
        "status": "AVAILABLE"
      }
    }
  },
  "mixed_device_models": false
}

All domain/channel/subset pairs and mechanism coverage are in public_aggregate.json. This descriptive assessment is not final external research selection. No automatic next experiment.

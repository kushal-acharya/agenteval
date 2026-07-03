"""AgentEval — eval + tracing harness for LLM agents.

Session map (see Notion spec):
  S1  dataset schema + adapter interface + serial runner  <- this scaffold
  S2  execution-based scorers (SQL result-set comparison)
  S3  LLM-as-judge + calibration
  S4  repeats/pass^k polish + `diff` command
  S5  tracing (Phoenix/OTel) + failure->trace links + CI regression gate
"""

__version__ = "0.1.0"

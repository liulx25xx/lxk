# 人工结果回传目录

当前状态：等待 24 位标注员全部返回。

不要将已完成的单人答案逐份上传到这里，否则后续标注员可能看到其他人的判断，破坏独立盲标。

全部返回后应先在本地完成：

1. raw 文件 SHA-256 冻结；
2. 2400 条 primary 与 240 条 QC 的完整性检查；
3. immutable 字段对比；
4. schema 校验；
5. 标注员之间的独立性确认。

确认所有标注工作结束后，再统一添加经过负责人批准的结果。建议结构：

```text
human_results/
  return_manifest.json
  raw/
    annotator_01/
      primary_100.jsonl
      qc_overlap_10.jsonl
    ...
  public_reports/
    validation_report.json
    agreement_summary.json
    disagreement_matrix.csv
    parser_error_by_domain_condition.csv
    corrected_accuracy_by_seed.csv
    corrected_accuracy_bootstrap.json
    public_summary.md
```

默认不要上传以下内容：

- 标注员真实姓名和联系方式；
- 本地 `private_do_not_share/`；
- 未聚合的 model/gold/parser join 文件；
- 仲裁者身份信息；
- 仍可能被其他标注员看到的中间答案。

完整接口与分析规则见：

`../docs/AI_HANDOFF_AND_HUMAN_RESULT_INTERFACE_ZH.md`

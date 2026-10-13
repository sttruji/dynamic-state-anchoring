#!/usr/bin/env python3
"""Validate and package individually written source-issue adjudications.

This command does not infer semantic verdicts or edit the corpus. Verdicts and
reasoning come from the completed reviewer ledger; source/coverage checks ensure
no flagged row is omitted, duplicated or associated with the wrong record.
"""
from pathlib import Path
from collections import Counter
from decimal import Decimal, InvalidOperation
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsa.data.adapters import canonical_record, stable_json_hash
from dsa.provenance import file_sha256

VERDICTS = {'correct', 'acceptable_approximation', 'incorrect', 'ambiguous', 'underdetermined', 'malformed_or_nonanswer'}


def plain_number(value):
    try:
        return Decimal(value.replace(',', '').strip()).is_finite()
    except InvalidOperation:
        return False


def validate_reviews(rows, reviews):
    expected = {i for i, row in enumerate(rows, 1) if not plain_number(row['answer'])}
    numbers = [record['row_number'] for record in reviews]
    if len(set(numbers)) != len(numbers) or set(numbers) != expected:
        raise ValueError('Adjudications must cover every unusual-answer row exactly once')
    result = []
    for record in sorted(reviews, key=lambda x: x['row_number']):
        source = rows[record['row_number'] - 1]
        canonical = canonical_record(source)
        if record.get('sample_id') is not None and record['sample_id'] != canonical['sample_id']:
            raise ValueError(f"Source identity mismatch for row {record['row_number']}")
        if record['verdict'] not in VERDICTS or record['confidence'] not in {'high', 'medium', 'low'}:
            raise ValueError('Unknown reviewer verdict or confidence')
        if not isinstance(record.get('reasoning'), str) or not record['reasoning'].strip():
            raise ValueError('Each semantic decision requires written reasoning')
        expected_handling = 'keep' if record['verdict'] in {'correct', 'acceptable_approximation'} else 'exclude_from_clean_sensitivity'
        if record['suggested_handling'] != expected_handling:
            raise ValueError('Sensitivity recommendation disagrees with verdict')
        answer_format = record['answer_format']
        if answer_format in {'approximation', 'approximate_decimal'}:
            answer_format = 'marked_approximation'
        result.append({**record, 'sample_id': canonical['sample_id'], 'answer_format': answer_format,
                       'source_answer': source['answer'], 'question': source['question'],
                       'rationale_raw': canonical['rationale_raw'],
                       'source_record_sha256': stable_json_hash(source),
                       'review_method': 'individual assistant semantic review and independent solution; contentious cases reread by primary reviewer'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', type=Path, default=ROOT/'artifacts/source-personal-review')
    args = parser.parse_args()
    source_path = ROOT/'data/gsm_train.json'
    before = file_sha256(source_path)
    rows = json.loads(source_path.read_text())
    reviews = [json.loads(line) for batch in (1, 2, 3)
               for line in (args.review_dir/f'adjudications_batch_{batch}.jsonl').open()]
    adjudicated = validate_reviews(rows, reviews)
    duplicates = [json.loads(line) for line in (args.review_dir/'duplicate_adjudications.jsonl').open()]
    for group in duplicates:
        for number in group['row_numbers']:
            if stable_json_hash(rows[number-1]) != group['source_record_sha256']:
                raise ValueError('Duplicate-group identity mismatch')
    if len(duplicates) != 52 or sum(len(g['row_numbers'])-1 for g in duplicates) != 60:
        raise ValueError('Incomplete duplicate review')
    conflict = json.loads((args.review_dir/'conflict_adjudication.json').read_text())
    conflict['source_records'] = [{'row_number':n, 'source_record_sha256':stable_json_hash(rows[n-1]), **rows[n-1]}
                                  for n in conflict['row_numbers']]
    ledger = ROOT/'provenance/lotus_personal_review_rows.jsonl'
    ledger.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in adjudicated))
    duplicate_ledger = ROOT/'provenance/lotus_personal_review_duplicates.jsonl'
    duplicate_ledger.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in duplicates))
    exclusions = sorted([r['row_number'] for r in adjudicated if r['suggested_handling']!='keep'] + conflict['row_numbers'])
    proposals = {'schema_version':'dsa_personal_review_sensitivity_proposal_v1',
                 'source_sha256':before,'applied':False,'row_number_convention':'1-based original training array order',
                 'proposed_quality_exclusion_row_numbers':exclusions,
                 'optional_duplicate_extra_row_numbers':sorted(n for group in duplicates for n in group['row_numbers'][1:]),
                 'primary_lotus_mirror_policy':'retain unchanged sources, annotations and duplicate exposure in all matched arms; disclose adjudicated noise',
                 'sensitivity_policy':'separately name and hash a derivative excluding adjudicated problematic rows, applying identical membership/order/exposure policy to every compared arm; do not silently replace the primary benchmark',
                 'note':'These are proposed row selections, not an edited or filtered dataset.'}
    proposal_path=ROOT/'provenance/lotus_personal_review_sensitivity_proposal.json'
    proposal_path.write_text(json.dumps(proposals, indent=2)+'\n')
    after=file_sha256(source_path)
    if after!=before:
        raise ValueError('Original source changed during packaging')
    summary={'schema_version':'dsa_lotus_personal_semantic_review_v1',
             'review_status':'completed_for_all_identified_answer_format_duplicate_and_conflict_issues',
             'reviewer':'assistant semantic review, authorized by the user; not represented as a human signature or certification',
             'source_path':str(source_path),'source_sha256':before,'source_unchanged':True,
             'training_rows_total':len(rows),'unusual_answer_rows_reviewed':len(adjudicated),
             'unusual_answer_verdict_counts':dict(sorted(Counter(r['verdict'] for r in adjudicated).items())),
             'answer_format_counts':dict(sorted(Counter(r['answer_format'] for r in adjudicated).items())),
             'acceptable_format_or_approximation_rows':sum(r['suggested_handling']=='keep' for r in adjudicated),
             'flagged_for_clean_sensitivity_rows':sum(r['suggested_handling']!='keep' for r in adjudicated),
             'duplicate_review':{'groups_reviewed':len(duplicates),'source_occurrences':sum(len(g['row_numbers']) for g in duplicates),'extra_occurrences':60,'extra_occurrence_fraction':60/len(rows),'correct_under_intended_word_problem_interpretation':52,'caveat':'Group12 uses an intended additive body-part model; the anatomical wording is awkward and noted in the ledger.','recommendation':'keep repeats in primary mirror; optional declared deduplicated sensitivity only'},
             'conflicting_question_review':conflict,
             'proposed_clean_sensitivity_exclusion_count':len(exclusions),
             'review_policy':{'exact_answers':'Check exact arithmetic, units, requested quantity and premises. Fractions, ratios, clock times, currency and intentionally symbolic answers can be valid.',
                              'approximations':'Explicit approximation markers with a sound method and appropriate magnitude can be accepted when precision is not specified; early-rounding bias is noted. Wrong required whole-object rounding or unmarked unequal exact fractions are not accepted.',
                              'ordinary_conventions':'Use ordinary word-problem conventions such as divisible food and sequential daily counts; state any material alternative assumptions.',
                              'ambiguity':'Do not guess omitted durations, counts, rates, compounding, bus wheel counts or incompatible premises.'},
             'next_step':'The user-authorized semantic source-issue review is complete. A small implementation pilot can use the unchanged primary mirror with documented noise; broad research conclusions still require controlled experiments and declared sensitivity analysis.',
             'remaining_limits':['This is a census of the194 unusual-answer flags,52 exact-duplicate groups and1 conflicting-question group, not semantic verification of every385620 training row.','Targeted issue-review proportions are not a corpus-wide error-rate estimate.','The existing32-record milestone sample was already reviewed by the assistant; useful latent checkpoints and generalization remain empirical questions.'],
             'artifacts':{str(p.relative_to(ROOT)):file_sha256(p) for p in (ledger, duplicate_ledger, proposal_path)},
             'assembly_script_sha256':file_sha256(Path(__file__))}
    (ROOT/'provenance/lotus_personal_review.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'reviewed':len(adjudicated),'verdicts':summary['unusual_answer_verdict_counts'],'duplicates':52,'proposed_sensitivity_exclusions':len(exclusions),'original_source_unchanged':True},indent=2))


if __name__=='__main__':
    main()

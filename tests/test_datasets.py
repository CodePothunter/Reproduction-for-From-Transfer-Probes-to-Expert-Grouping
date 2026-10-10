"""离线契约测试；字符 tokenizer 仅作测试替身，不是生产 token 计数。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from lwts.datasets import DEEPMATH, prepare_records, split_records, validate_manifest
from lwts.storage import load, save

VERSION = 'main'


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {'input_ids': list(range(len(text))),
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}


def sample(question='Compute 1+1', topic='Mathematics -> Algebra'):
    return dict(question=question, topic=topic, final_answer='2',
                r1_solution_1='1+1=2', r1_solution_2='two', r1_solution_3='')


class DataTests(unittest.TestCase):
    def prepare(self, rows, **kwargs):
        return prepare_records(rows, Tokenizer(), revision=VERSION, **kwargs)

    def manifest(self):
        rows = self.prepare([sample()])
        rows = split_records(rows, dict(train=1, feedback=0, route=0, audit=0, test=0), 'seed')
        return dict(schema_version=1, dataset_revisions={DEEPMATH: VERSION},
                    tokenizer={'id': 'test', 'revision': VERSION}, sources=[0],
                    split_salt='seed', required_splits=['train'], records=rows,
                    status="draft")

    def test_selection_reference_and_dedup_order(self):
        a, b = sample(), sample()
        a['unique_id'], b['unique_id'] = 'a', 'b'
        b['r1_solution_1'] = 'different'
        report = {}
        rows = self.prepare([a, b], report=report)
        self.assertEqual(rows, self.prepare([b, a]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['reference']['answer'], '2')
        self.assertEqual(report['reasons'], {'duplicate': 1})
        self.assertNotIn('final_answer', rows[0]['prompt'])

    def test_validator_fallback_and_length(self):
        rows = self.prepare([sample()], target_validator=lambda text, ref: text == 'two')
        self.assertEqual(rows[0]['metadata']['target_choice'], 'r1_solution_2')
        self.assertEqual(rows[0]['completion_tokens'], 3)
        self.assertEqual(self.prepare([sample()], max_tokens=1), [])

    def test_topic_filter_and_conflict(self):
        report = {}
        self.assertEqual(self.prepare([sample(topic='Geometry')], report=report), [])
        self.assertEqual(report['reasons'], {'out_of_scope': 1})
        with self.assertRaises(ValueError):
            self.prepare([sample(), sample(topic='Number Theory')])

    def test_split_stable_nonmutating(self):
        rows = self.prepare([sample(str(i)) for i in range(30)])
        original = copy.deepcopy(rows)
        ratios = dict(train=.6, feedback=.1, route=.1, audit=.1, test=.1)
        a = split_records(rows, ratios, 'salt')
        b = split_records(list(reversed(rows)), ratios, 'salt')
        self.assertEqual({r['id']: r['split'] for r in a}, {r['id']: r['split'] for r in b})
        a[0]['metadata']['topic'] = 'changed'
        self.assertEqual(rows, original)

    def test_official_test_and_bad_ratios(self):
        rows = self.prepare([sample()], original_split='test')
        with self.assertRaises(ValueError):
            split_records(rows, dict.fromkeys(('train', 'feedback', 'route', 'audit', 'test'), .2), 's')
        with self.assertRaises(ValueError):
            split_records([], dict(train=float('nan'), feedback=0, route=0, audit=0, test=0), 's')

    def test_manifest_integrity_leakage_and_freeze(self):
        m = self.manifest()
        self.assertEqual(validate_manifest(m)['records'], 1)
        m['records'][0]['completion_tokens'] = m['records'][0]['tokens'] + 1
        with self.assertRaises(ValueError):
            validate_manifest(m)
        m = self.manifest()
        row = copy.deepcopy(m['records'][0])
        row.update(id='another', split='test')
        m['records'].append(row)
        with self.assertRaises(ValueError):
            validate_manifest(m)
        m = self.manifest()
        m.update(status='frozen', external_audit='passed')
        self.assertEqual(validate_manifest(m)['records'], 1)

    def test_missing_source_and_revision(self):
        m = self.manifest()
        m['sources'].append(1)
        with self.assertRaises(ValueError):
            validate_manifest(m)
        m = self.manifest()
        m['dataset_revisions'][DEEPMATH] = ''
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_math_prompt_template_is_shared_and_carries_the_instruction(self):
        """数学来源的每个切分都走同一个模板：训练用的与评估用的输入逐字相同。

        两处各写一份字面量时靠巧合保持一致；一旦分叉，模型就在 A 格式下训练、
        在 B 格式下被评估，而两边各自都自洽，**没有任何检查会发现**。
        """
        from lwts.datasets import DEEPMATH
        from lwts.datasets.common import math_prompt
        question = 'Compute 2+2'
        row = dict(question=question, final_answer='4', topic='Algebra',
                   r1_solution_1='the answer is 4')
        train = prepare_records([row], Tokenizer(), dataset=DEEPMATH, original_split='train')
        test = prepare_records([row], Tokenizer(), dataset=DEEPMATH, original_split='test')
        self.assertEqual(train[0]['prompt'], test[0]['prompt'])
        self.assertEqual(train[0]['prompt'], math_prompt(question))
        # 官方指令句必须在里面：它是实验前定下的模板决策，不能只停留在文档里。
        self.assertIn('within \\boxed{}.', train[0]['prompt'])
        self.assertTrue(train[0]['prompt'].endswith('\n\nSolution:\n'))

    def test_opencodeinstruct_sources_and_reference_quality(self):
        """按 domain 分源，并排掉参考解通不过自身单测的题目。

        实测只有 25-46% 的条目参考解合格。不过滤就会把「题目本身有问题」记成
        「模型答错」，给代码来源的 competence 加一份与模型无关的负偏差。
        """
        from lwts.datasets import OPENCODEINSTRUCT
        base = dict(output='```python\ndef add(a, b):\n    return a + b\n```',
                    unit_tests="['assert add(1, 2) == 3']", average_test_score=1.0)
        rows = [dict(base, input='Implement add.', domain='algorithmic', id='a'),
                dict(base, input='Parse dates.', domain='generic', id='b'),
                dict(base, input='Bad reference.', domain='generic',
                     average_test_score=0.8, id='c'),
                dict(base, input='Unknown domain.', domain='other', id='d')]
        stats = {}
        out = prepare_records(rows, Tokenizer(), dataset=OPENCODEINSTRUCT,
                              original_split='train', report=stats)
        self.assertEqual([r['source'] for r in out], [2, 3])
        self.assertEqual(sorted(e['reason'] for e in stats['excluded']),
                         ['out_of_scope', 'reference_fails_own_tests'])
        self.assertEqual(out[0]['reference']['kind'], 'assertions')
        self.assertEqual(out[0]['reference']['tests'], ['assert add(1, 2) == 3'])

    def test_opencodeinstruct_default_shards_cover_the_rare_domain(self):
        """domain 呈块状分布：只读头部分片会一条 algorithmic 都取不到。"""
        from lwts.datasets import opencodeinstruct as oci
        self.assertIn(45, oci.DEFAULT_SHARDS)
        files = oci.data_files()
        self.assertEqual(len(files), len(oci.DEFAULT_SHARDS))
        self.assertTrue(all(f.endswith(f'-of-{oci.SHARD_COUNT:05d}.parquet')
                            for f in files))
        with self.assertRaises(ValueError):
            oci.data_files(shards=(oci.SHARD_COUNT,))

    def test_opencodeinstruct_drops_conflicting_domain_labels(self):
        """同题被标成两个 domain 时整题排除，而不是任选一个。

        实测这类冲突占 0.23%（246/107332）。任选会给两个来源都注入标签噪声；
        不清理则会撞上数据层「同题被分配到不同来源」的硬中止。
        """
        from lwts.datasets import opencodeinstruct as oci

        class FakeData:
            """只实现 drop_domain_conflicts 用到的两个接口。"""

            def __init__(self, rows):
                self.rows = rows

            def __getitem__(self, column):
                return [r[column] for r in self.rows]

            def filter(self, predicate):
                return FakeData([r for r in self.rows if predicate(r)])

        data = FakeData([
            {"input": "Sort a list.", "domain": "generic"},
            {"input": "Sort  a list.", "domain": "algorithmic"},   # 规范化后同题
            {"input": "Reverse a string.", "domain": "generic"},
        ])
        kept = oci.drop_domain_conflicts(data)
        self.assertEqual([r["input"] for r in kept.rows], ["Reverse a string."])
        # 没有冲突时原样返回，不做多余的一遍扫描
        clean = FakeData([{"input": "Reverse a string.", "domain": "generic"}])
        self.assertIs(oci.drop_domain_conflicts(clean), clean)

    def test_escaped_surrogate_pairs_are_rejoined_not_dropped(self):
        """literal_eval 会把一个 emoji 拆成两个落单代理，必须合并回去。

        OpenCodeInstruct 有 590/300000 行的 unit_tests 中招，全部成对。
        `\\ud83d\\ude0a` 是 U+1F60A 😊 的 UTF-16 编码，其 UTF-8 字节 base64 后
        正是 "8J+Yig=="——合并后单测成立，不合并则 .encode() 抛异常，模型写
        什么都过不了，等于给代码来源注入一批与模型无关的零奖励。
        """
        from lwts.datasets import OPENCODEINSTRUCT
        from lwts.datasets.common import parse_field, repair_surrogates
        emoji = '\U0001f60a'
        # 显式拼代理对：直接在源码里写 emoji 得到的是完好字符，测不出任何东西。
        pair = chr(0xd83d) + chr(0xde0a)
        self.assertEqual(len(pair), 2)
        self.assertRaises(UnicodeEncodeError, pair.encode, 'utf-8')
        broken = f'assert encode_message("{pair}") == "8J+Yig=="'
        self.assertEqual(repair_surrogates(broken),
                         f'assert encode_message("{emoji}") == "8J+Yig=="')
        # 校验这个断言本身是对的：合并后的字符 UTF-8 编码 base64 确实是 8J+Yig==
        import base64
        self.assertEqual(base64.b64encode(emoji.encode()).decode(), '8J+Yig==')
        # 未中招的文本原样返回，不做无谓的 utf-16 往返
        clean = 'assert add(1, 2) == 3'
        self.assertIs(repair_surrogates(clean), clean)
        self.assertEqual(parse_field(repr([broken]), list, literal=True),
                         [f'assert encode_message("{emoji}") == "8J+Yig=="'])

        row = dict(input='Encode a message.', domain='algorithmic',
                   output='```python\ndef encode_message(s):\n    return s\n```',
                   unit_tests=repr([broken]), average_test_score=1.0, id='a')
        out = prepare_records([row], Tokenizer(), dataset=OPENCODEINSTRUCT,
                              original_split='train')
        self.assertEqual(out[0]['reference']['tests'],
                         [f'assert encode_message("{emoji}") == "8J+Yig=="'])
        # 能 UTF-8 落盘是硬要求：manifest 是 JSONL，写不出去整批就废了
        json.dumps(out[0], ensure_ascii=False).encode('utf-8')

    def test_lone_surrogate_is_excluded_not_a_crash(self):
        """落单代理没有合法 UTF-8 编码，按条排除并给出可读原因。

        兜底防线：不设它的话，json.dumps 会在所有解析和 bank 采样都做完之后
        的落盘阶段整批崩掉，报错还指不出是哪个数据集哪一条。
        """
        from lwts.datasets import OPENCODEINSTRUCT
        from lwts.datasets.common import repair_surrogates
        with self.assertRaises(ValueError):
            repair_surrogates('lone ' + chr(0xd83d) + ' here')

        base = dict(output='```python\ndef f(s):\n    return s\n```',
                    average_test_score=1.0)
        # 单测字段存的是 Python repr，代理写成转义；literal_eval 解出落单代理。
        lone = r'[\'assert f("\ud83d") == 1\']'
        rows = [dict(base, input='Good one.', domain='generic',
                     unit_tests="['assert f(1) == 1']", id='a'),
                dict(base, input='Lone surrogate.', domain='generic',
                     unit_tests=lone, id='b')]
        stats = {}
        out = prepare_records(rows, Tokenizer(), dataset=OPENCODEINSTRUCT,
                              original_split='train', report=stats)
        self.assertEqual([r['reference']['tests'] for r in out],
                         [['assert f(1) == 1']])
        self.assertEqual([e['reason'] for e in stats['excluded']],
                         ['unparsable_unit_tests'])

    def test_mbppplus_is_external_only_and_shows_the_signature(self):
        """题面只有一句话描述、不含函数名，必须附断言模型才知道签名。"""
        from lwts.datasets import MBPPPLUS
        row = dict(task_id=2, prompt='Write a function to add two numbers.',
                   test_list=['assert add(1, 2) == 3'],
                   test='assert add(2, 2) == 4', test_imports=[],
                   code='def add(a, b): return a + b')
        out = prepare_records([row], Tokenizer(), dataset=MBPPPLUS,
                              topic_sources={'code': 2})
        self.assertTrue(out[0]['metadata']['external'])
        self.assertEqual(out[0]['completion_tokens'], 0)
        self.assertIn('assert add(1, 2) == 3', out[0]['prompt'])
        # 判分用 EvalPlus 增强后的 test，不是基础的 test_list
        self.assertEqual(out[0]['reference']['tests'], ['assert add(2, 2) == 4'])
        with self.assertRaises(ValueError):
            prepare_records([row], Tokenizer(), dataset=MBPPPLUS,
                            topic_sources={'code': 2}, original_split='train')

    def test_code_train_and_external_share_one_prompt_wrapper(self):
        """与数学侧同理：训练与最终评估的输入包装必须由单一来源保证。"""
        from lwts.datasets import MBPPPLUS, OPENCODEINSTRUCT
        train = prepare_records(
            [dict(input='Implement add.', domain='generic', average_test_score=1.0,
                  unit_tests="['assert add(1, 2) == 3']",
                  output='```python\ndef add(a, b):\n    return a + b\n```')],
            Tokenizer(), dataset=OPENCODEINSTRUCT, original_split='train')
        external = prepare_records(
            [dict(task_id=1, prompt='Implement add.', test_list=['assert add(1, 2) == 3'],
                  test='assert add(2, 2) == 4', test_imports=[], code='x')],
            Tokenizer(), dataset=MBPPPLUS, topic_sources={'code': 3})
        suffix = '\n\nPython solution:\n'
        self.assertTrue(train[0]['prompt'].endswith(suffix))
        self.assertTrue(external[0]['prompt'].endswith(suffix))

    def test_same_problem_different_reference_is_a_duplicate(self):
        """真实语料里同题带两套测试集是常态，按重复处理并单列原因。"""
        from lwts.datasets import OPENCODEINSTRUCT
        base = dict(input='Implement add.', domain='generic', average_test_score=1.0,
                    output='```python\ndef add(a, b):\n    return a + b\n```')
        a = dict(base, unit_tests="['assert add(1, 2) == 3']", id='p1')
        b = dict(base, unit_tests="['assert add(1, 2) == 3', 'assert add(0, 0) == 0']",
                 id='p2')
        report = {}
        rows = prepare_records([a, b], Tokenizer(), dataset=OPENCODEINSTRUCT,
                               original_split='train', report=report)
        self.assertEqual(len(rows), 1)
        self.assertEqual(report['reasons'], {'duplicate_conflicting_reference': 1})

    def test_same_problem_different_source_still_aborts(self):
        """同题跨来源会破坏来源纯净性，必须中止而不是排除。"""
        with self.assertRaises(ValueError):
            self.prepare([sample(), sample(topic='Number Theory')])

    def test_empty_target_list_has_a_readable_reason(self):
        """原始记录一个参考解都没给时，排除原因要可读，而不是混进别的类别。"""
        from lwts.datasets.common import Excluded, Parsed, _select_target
        parsed = Parsed(source=0, identity='Q', prompt='Q', reference={'kind': 'x'},
                        targets=[], topic=None)
        with self.assertRaises(Excluded) as caught:
            _select_target(parsed, Tokenizer(), None, None)
        self.assertEqual(caught.exception.reason, 'no_target')
        empty = parsed._replace(targets=[('output', '  ')])
        with self.assertRaises(Excluded) as caught:
            _select_target(empty, Tokenizer(), None, None)
        self.assertEqual(caught.exception.reason, 'no_eligible_target:empty_target')

    def test_unified_input_and_registry(self):
        """未注册的 dataset 走统一输入；dataset 只作版本标签。"""
        from lwts.datasets import DATASETS, MODULES
        rows = prepare_records(
            [dict(prompt='P:', problem='my problem', source=9,
                  reference={'kind': 'x'}, completion='done')],
            Tokenizer(), dataset='my-corpus')
        self.assertEqual(rows[0]['source'], 9)
        self.assertEqual(rows[0]['metadata']['dataset'], 'my-corpus')
        self.assertEqual(set(DATASETS), set(MODULES))
        for name, module in MODULES.items():
            self.assertTrue(module.SPEC.dataset, name)

    def test_plain_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'state.json'
            save(path, {'version': 'main', 'records': []})
            self.assertEqual(load(path), {'version': 'main', 'records': []})


if __name__ == '__main__':
    unittest.main()

import gzip
import tempfile
import unittest
from pathlib import Path
from prepare_data import prepare_graph,prepare_matrix

test_root=(Path(__file__).resolve().parent/'tmp-tests').resolve()
assert test_root.is_relative_to(Path(__file__).resolve().parents[2])
test_root.mkdir(exist_ok=True)
tempfile.tempdir=str(test_root)

class DataPreparationTests(unittest.TestCase):
    def test_undirected_dedup_and_root_component(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'g.gz';dst=root/'g.csr'
            with gzip.open(src,'wt') as f:f.write('# graph\n10 20\n20 10\n20 30\n40 50\n')
            meta=prepare_graph(src,dst)
            self.assertEqual(meta['rows'],5)
            self.assertEqual(meta['stored_directed_edges'],6)
            self.assertEqual(meta['root_original_id'],20)
            self.assertEqual(meta['reachable'],3)
            self.assertEqual(meta['max_distance'],1)
            lines=dst.read_text().splitlines()
            self.assertEqual(lines[0],'PCSR1 5 5 6')
            self.assertEqual(list(map(int,lines[1].split()))[:2],[0,2])

    def test_symmetric_expansion_without_doubling_diagonal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'m.gz';dst=root/'m.csr'
            with gzip.open(src,'wt') as f:
                f.write('%%MatrixMarket matrix coordinate real symmetric\n% test\n3 3 4\n1 1 2\n2 1 3\n2 2 4\n3 3 5\n')
            meta=prepare_matrix(src,dst)
            self.assertEqual(meta['expanded_nnz'],5)
            lines=dst.read_text().splitlines()
            self.assertEqual(list(map(int,lines[1].split())),[0,2,4,5])
            self.assertEqual(list(map(int,lines[2].split())),[0,1,0,1,2])
            self.assertEqual(list(map(float,lines[3].split())),[2,3,3,4,5])

    def test_truncated_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'m.gz'
            with gzip.open(src,'wt') as f:
                f.write('%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 1 1\n')
            with self.assertRaisesRegex(ValueError,'entry count'):
                prepare_matrix(src,root/'out.csr')

if __name__=='__main__':unittest.main()

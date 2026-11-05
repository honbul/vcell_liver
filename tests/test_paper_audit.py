"""A restarted attempt may bridge only a verified checkpoint at the log boundary."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paper_audit import audit


class PaperAuditTest(unittest.TestCase):
    def test_interrupted_checkpoint_print_and_resumed_final_export(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, restarted = root/'first.log', root/'second.log'
            first.write_text('EPOCH_CHECKPOINT 1 0.3\n')
            restarted.write_text('EPOCH_CHECKPOINT 3 0.2\nR1_FINISHED output\n')
            receipt = dict(epoch=2,selection_value=.25,checkpoint_sha256='fixture',
                           receipt_sha256='fixture')
            with self.assertRaisesRegex(ValueError,'Unexpected epoch checkpoint'):
                audit(restarted,'R1_FINISHED',3)
            combined = audit(first,'R1_FINISHED',3,[(restarted,receipt)])
            self.assertEqual([r['epoch'] for r in combined['checkpoints']],[1,2,3])
            self.assertEqual(combined['logs'][1]['resumed_from']['epoch'],2)
            restarted.write_text('EPOCH_CHECKPOINT 2 0.2\nR1_FINISHED output\n')
            with self.assertRaisesRegex(ValueError,'Unexpected epoch checkpoint'):
                audit(first,'R1_FINISHED',3,[(restarted,receipt)])


if __name__ == '__main__':
    unittest.main()

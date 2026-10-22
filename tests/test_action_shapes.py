"""Keep format failures separate from syntactically valid semantic failures."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_reflexion_action_shapes import action_shape
from analyze_reflexion_pilot_forks import object_type


class ActionShapeTests(unittest.TestCase):
    def test_public_two_object_goals_extract_the_object(self):
        for goal, expected in [('find two bowl and put them in desk.', 'bowl'),
                               ('put two ladle in drawer.', 'ladle'),
                               ('clean some spatula and put it in diningtable.', 'spatula'),
                               ('put a cloth in drawer.', 'cloth')]:
            with self.subTest(goal=goal):
                self.assertEqual(object_type(goal), expected)

    def test_role_echo_and_semantic_mistakes_have_different_labels(self):
        self.assertEqual(action_shape('You open the fridge 1.', 'You open the fridge 1.'),
                         'environment_style_sentence')
        # A handtowel can be the wrong goal object while remaining a legal command.
        self.assertEqual(action_shape('put handtowel 1 in drawer 1', 'move handtowel 1 to drawer 1'),
                         'public_command_shape')
        self.assertEqual(action_shape('open fridge 99', 'open fridge 99'), 'public_command_shape')
        self.assertEqual(action_shape('think: search again', 'think: search again'), 'thought')
        self.assertEqual(action_shape('look around desk 1', 'look around desk 1'), 'other_unmatched_shape')


if __name__ == '__main__':
    unittest.main()

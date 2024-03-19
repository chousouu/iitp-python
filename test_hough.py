from fasthough import HoughTransform


class TestClass:
    def test_empty(self):
        fht = HoughTransform("tests/empty.png")
        assert len(fht.get_lines()) == 0

    def test_empty_rect(self):
        fht = HoughTransform("tests/empty_rect.png")
        assert len(fht.get_lines()) == 0

    def test_horizontal_line(self):
        fht = HoughTransform("tests/horizontal_line.png")
        assert len(fht.get_lines()) == 1

    def test_vertical_line(self):
        fht = HoughTransform("tests/vertical_line.png")
        assert len(fht.get_lines()) == 1

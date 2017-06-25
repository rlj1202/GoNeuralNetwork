package mat

import "testing"

func TestRemap(t *testing.T) {
	t.Log(Remap(2015, 2010, 2020, -1, 0))
	t.Log(Remap(0.5, 0, 1, 50, 100))
}
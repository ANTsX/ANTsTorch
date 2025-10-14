import unittest
import ants
import antstorch

class Test_t1(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_example(self):
        t1 = ants.image_read(antstorch.get_antsxnet_data('mprage_hippmapp3r'))
        seg = antstorch.brain_extraction(t1, modality="t1")

# class Test_t1threetissue(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antstorch.get_antsxnet_data('mprage_hippmapp3r'))
#         bext = antstorch.brain_extraction(t1, modality="t1threetissue")

# class Test_t1hemi(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antstorch.get_antsxnet_data('mprage_hippmapp3r'))
#         bext = antstorch.brain_extraction(t1, modality="t1hemi")

# class Test_t1lobes(unittest.TestCase):
#     def setUp(self):
#         pass
#     def tearDown(self):
#         pass
#     def test_example(self):
#         t1 = ants.image_read(antstorch.get_antsxnet_data('mprage_hippmapp3r'))
#         bext = antstorch.brain_extraction(t1, modality="t1lobes")

if __name__ == '__main__':
    unittest.main()
import unittest

from Model.Cut import Cut
from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire
from Model.DataType import DataType
from Model.SearchTree import condense_tree, hard_clustering, contracting_search_tree, Searchtree, print_tree, soft_clustering
from Model.TangleCluster import consistent, create_searchtree  # Update with your actual import path
from Model.GenerateTestData import GenerateDataBinaryQuestionnaire

class TestClusteringFunctions(unittest.TestCase):
    def setUp(self):
        # Setup your mock data and dependencies here
        self.number_of_clusters = 4
        self.number_of_questions = 100
        self.number_of_participants = 100
        self.agreement_parameter = 13
        self.sim_fun = "Element by element"

        self.generated_data = GenerateDataBinaryQuestionnaire(self.number_of_participants, self.number_of_questions, self.number_of_clusters)
        self.generated_data.generate_biased_binary_questionnaire_answers()
        self.generated_data.res_to_points("t-SNE")
       
        data = DataSetBinaryQuestionnaire(self.agreement_parameter).cut_generator_binary(self.generated_data.questionaire, self.sim_fun)        

        self.root = create_searchtree(data)
       
        
        self.condensed_tree = condense_tree(self.root)

        
        contracting_search_tree(self.condensed_tree)
        self.soft_clustering = soft_clustering(self.condensed_tree)
        self.hard_clustering = hard_clustering(self.soft_clustering)

    def test_hard_clustering(self):
        # Assuming hard_clustering returns a list of cluster labels
        clusters = hard_clustering(self.soft_clustering)
        # Assert that the clustering output is correct
        self.assertIsInstance(clusters, list)  # Further checks as needed
        self.assertEqual(len(set(clusters)), self.number_of_clusters)

    def test_condense_tree(self):
        # Setup a test tree structure for the condense_tree function
       
        root = Searchtree(None, '0')

        child1 = Searchtree(root, '1')
        child2 = Searchtree(root, '2')
        leaf1 = Searchtree(child1, '3')
        leaf2 = Searchtree(child2, '4')
        
        root.add_left_child(child1)
        root.add_right_child(child2)
        child1.add_left_child(leaf1)
        child2.add_right_child(leaf2)

        
        self.assertEqual(root.left_node.cut_id, child1.cut_id, "Leaf1 should replace Child1")
        self.assertEqual(root.right_node.cut_id, child2.cut_id, "Leaf2 should replace Child2")
        # Conduct the test
        new_root = condense_tree(root)
    
        self.assertEqual(new_root.left_node.cut_id, leaf1.cut_id, "Leaf1 should replace Child1")
        self.assertEqual(new_root.right_node.cut_id, leaf2.cut_id, "Leaf2 should replace Child2")

    def test_consistent(self):
        # Assuming consistent returns a boolean
        
        A = {1, 2, 3}
        agreement_parameter_1 = 3

        node1 = Searchtree(None, '0')
        node1.tangle = [[{1, 2, 3, 4}], [{1, 2, 3}]]

        agreement_parameter_2 = 4
        node2 = Searchtree(None, '0')
        node2.tangle = [[{1, 2, 3}]]
      
        self.assertTrue(consistent(A, node1, agreement_parameter_1))
        self.assertFalse(consistent(A, node2, agreement_parameter_2))



    def test_consruct_search_tree(self):
        cut1 = Cut(1, 0.5, "L", {1, 2, 3}, {4, 5, 6})
        # cut2 = Cut(2, 0.5, "R", {7, 8, 9}, {10, 11, 12})
        cuts = [cut1]

        agreement_param = 3

        data = DataSetBinaryQuestionnaire(agreement_param, cuts)

        root = create_searchtree(data)

        self.assertEqual(root.cut_id, 0)
        self.assertEqual(root.left_node.cut_id, 1)
        self.assertEqual(root.right_node.cut_id, 1)
        
        

    def test_contract_tree(self):
        """
            tree looks like this:
               0       
              / \
             1   2

        """
        root = Searchtree(None, '0')

        leaf1 = Searchtree(root, '1')
        leaf2 = Searchtree(root, '2')

        root.add_left_child(leaf1)
        root.add_right_child(leaf2)

        new_root = condense_tree(root)
        contracting_search_tree(new_root)

        

        


if __name__ == '__main__':
    unittest.main()

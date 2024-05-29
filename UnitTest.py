import unittest

from Model.DataSetBinaryQuestionnaire import DataSetBinaryQuestionnaire
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
        print_tree(self.root)
        
        self.condensed_tree = condense_tree(self.root)

        print_tree(self.condensed_tree)

        print_tree(self.condensed_tree)
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
        root = Searchtree(None, 'root')
        child1 = Searchtree(root, '1')
        child2 = Searchtree(root, '2')
        leaf1 = Searchtree(child1, '3')
        leaf2 = Searchtree(child2, '4')

        root.left_node = child1
        root.right_node = child2
        child1.left_node = leaf1
        child2.right_node = leaf2

        
        # print_tree(root)
        # Conduct the test
        new_root = condense_tree(root)

        # print_tree(new_root)

    
        self.assertEqual(new_root.left_node.cut_id, leaf1.cut_id, "Child1 should be detached if conditions met.")
        self.assertEqual(new_root.right_node.cut_id, leaf2.cut_id, "Child2 should remain attached if not meeting conditions.")
        

        

if __name__ == '__main__':
    unittest.main()

from typing import List, Dict, Optional, Any

class DomainTree:
    """
    A tree structure utility class for caching and managing hierarchical labels.
    Supports node addition, deletion, update, search, path finding, and tree serialization......
    """
    def __init__(self, tree_data: Optional[List[Dict[str, Any]]] = None) -> None:
        self.tree: List[Dict[str, Any]] = tree_data or []
        
    #add node
    def add_node(self, label: str, parent_label: Optional[str] = None) -> bool:
        if parent_label is None:
            self.tree.append({"label": label})
            return True
        parent = self.find_node(parent_label)
        if parent is not None:
            if "child" not in parent:
                parent["child"] = []
            parent["child"].append({"label": label})
            return True
        return False

    #remove node
    def remove_node(self, label: str) -> bool:
        for index, node in enumerate(self.tree):
            if node.get("label") == label:
                del self.tree[index]
                return True
        for node in self.tree:
            if "child" in node:
                for child_index, child in enumerate(node["child"]):
                    if child.get("label") == label:
                        del node["child"][child_index]
                        return True
        return False

    #update node
    def update_node(self, old_label: str, new_label: str) -> bool:
        node = self.find_node(old_label)
        if node:
            node["label"] = new_label
            return True
        return False

    #find node
    def find_node(self, label: str) -> Optional[Dict[str, Any]]:
        for node in self.tree:
            if node.get("label") == label:
                return node
            if "child" in node:
                for child in node["child"]:
                    if child.get("label") == label:
                        return child
        return None

    #find path
    def find_path(self, label: str) -> Optional[str]:
        #recursive method need more parameters
        def _find(tree: List[Dict[str, Any]], label: str, path: List[str]) -> Optional[List[str]]:
            for node in tree:
                current_path = path + [node.get("label", "")]
                if node.get("label") == label:
                    return current_path
                if "child" in node:
                    result = _find(node["child"], label, current_path)
                    if result:
                        return result
            return None
        result = _find(self.tree, label, [])
        return "/".join(result) if result else None

        
    def to_json(self) -> List[Dict[str, Any]]:
        return self.tree

    def from_json(self, json_data: List[Dict[str, Any]]) -> None:
        self.tree = json_data


if __name__ == "__main__":
    import json
    print("[DomainTree Main Interface Demo]")
    # 1. Create empty tree
    tree = DomainTree()
    print("***1. Empty tree: Create an empty tree structure.***", tree.to_json())

    # 2. Add root nodes
    tree.add_node("Computer Science")
    tree.add_node("Mathematics")
    print("***2. Add root nodes: Add root nodes 'Computer Science' and 'Mathematics'.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 3. Add child nodes
    tree.add_node("Artificial Intelligence", "Computer Science")
    tree.add_node("Machine Learning", "Artificial Intelligence")
    tree.add_node("Deep Learning", "Machine Learning")
    print("***3. Add child nodes: Add child nodes under the corresponding parent nodes.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 4. Find node
    node = tree.find_node("Machine Learning")
    print("***4. Find node: Find node 'Machine Learning', if not exist return None.***", node)

    # 5. Find path
    path = tree.find_path("Deep Learning")
    print("***5. Find path: Find the path of node 'Deep Learning', return path string or None.***", path)

    # 6. Update node
    tree.update_node("Machine Learning", "ML")
    print("***6. Update node: Update node label from 'Machine Learning' to 'ML'.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 7. Remove node
    tree.remove_node("ML")
    print("***7. Remove node: Remove node 'ML', if not exist do nothing.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 8. Export as JSON
    print("***8. Export as JSON: Export the tree structure as a JSON-serializable list.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 9. Import from JSON
    json_data = [
        {"label": "Technology", "child": [
            {"label": "Programming Language", "child": [
                {"label": "Python"}, {"label": "Java"}
            ]}
        ]}
    ]
    tree.from_json(json_data)
    print("9. Import from JSON:", json.dumps(tree.to_json(), ensure_ascii=False, indent=2)) 
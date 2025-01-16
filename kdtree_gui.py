import tkinter as tk
from tkinter import ttk, messagebox
from kdtree import kdtree_main


class KDTreeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KD-Tree Search GUI")

        # Open the window in front of other apps
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)

        # Open the window in full window mode (maximized)
        self.root.state('zoomed')

        # Available attributes
        self.available_columns = ["100g_USD", "rating", "review_date", "roaster", "roast", "loc_country", "origin"]

        # Numeric attributes (these will have min and max entry fields)
        self.numeric_attributes = ["100g_USD", "rating", "review_date"]

        # Selected attributes and conditions
        self.selected_attributes = []
        self.conditions = {}

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        attr_frame = ttk.LabelFrame(self.root, text="Select attributes (up to 4)")
        attr_frame.pack(fill="x", padx=10, pady=10)

        self.attr_vars = {}
        for i, col in enumerate(self.available_columns):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(attr_frame, text=col, variable=var, command=lambda c=col: self.update_conditions(c))
            chk.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=2)
            self.attr_vars[col] = var

        self.review_var = tk.BooleanVar()
        review_chk = ttk.Checkbutton(attr_frame, text="review", variable=self.review_var,
                                     command=lambda: self.update_conditions("review"))
        review_chk.grid(row=len(self.available_columns) // 2, column=len(self.available_columns) % 2, sticky="w",
                        padx=5, pady=2)

        self.cond_frame = ttk.LabelFrame(self.root, text="Enter conditions. For numeric attributes enter min and max values. For non-numeric attributes, use commas. Dates are in YYYYMM format.")
        self.cond_frame.pack(fill="x", padx=10, pady=10)

        self.cond_labels = {}
        self.min_entries = {}
        self.max_entries = {}
        self.single_entries = {}
        for i, col in enumerate(self.available_columns):
            label = ttk.Label(self.cond_frame, text=f"{col}:")
            label.grid(row=i, column=0, padx=5, pady=2, sticky="e")

            if col in self.numeric_attributes:
                min_label = ttk.Label(self.cond_frame, text="Min:")
                min_label.grid(row=i, column=1, padx=(5, 0), pady=2, sticky="e")

                min_entry = ttk.Entry(self.cond_frame, width=10)
                min_entry.grid(row=i, column=2, padx=(0, 5), pady=2, sticky="w")

                max_label = ttk.Label(self.cond_frame, text="Max:")
                max_label.grid(row=i, column=3, padx=(5, 0), pady=2, sticky="e")

                max_entry = ttk.Entry(self.cond_frame, width=10)
                max_entry.grid(row=i, column=4, padx=(0, 5), pady=2, sticky="w")

                self.min_entries[col] = min_entry
                self.max_entries[col] = max_entry
            else:
                single_entry = ttk.Entry(self.cond_frame, width=20)
                single_entry.grid(row=i, column=1, padx=5, pady=2, sticky="w", columnspan=4)

                self.single_entries[col] = single_entry

            self.cond_labels[col] = label
            label.grid_remove()
            if col in self.numeric_attributes:
                min_label.grid_remove()
                min_entry.grid_remove()
                max_label.grid_remove()
                max_entry.grid_remove()
            else:
                single_entry.grid_remove()

        self.review_label = ttk.Label(self.cond_frame, text="Review Keywords:")
        self.review_label.grid(row=len(self.available_columns), column=0, padx=5, pady=2, sticky="e")

        self.review_entry = ttk.Entry(self.cond_frame, width=20)
        self.review_entry.grid(row=len(self.available_columns), column=1, padx=5, pady=2, sticky="w", columnspan=4)

        self.num_neighbors_label = ttk.Label(self.cond_frame, text="Number of Nearest Neighbors:")
        self.num_neighbors_label.grid(row=len(self.available_columns) + 1, column=0, padx=5, pady=2, sticky="e")

        self.num_neighbors_entry = ttk.Entry(self.cond_frame, width=10)
        self.num_neighbors_entry.grid(row=len(self.available_columns) + 1, column=1, padx=5, pady=2, sticky="w")

        self.review_label.grid_remove()
        self.review_entry.grid_remove()
        self.num_neighbors_label.grid_remove()
        self.num_neighbors_entry.grid_remove()

        search_btn = ttk.Button(self.root, text="Search", command=self.perform_search)
        search_btn.pack(pady=10)

        self.results_frame = ttk.LabelFrame(self.root, text="Search results. Tap on the headings to sort the table.")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        tree_container = ttk.Frame(self.results_frame)
        tree_container.pack(fill="both", expand=True)

        self.results_tree = ttk.Treeview(
            tree_container,
            columns=("Name", "Roaster", "Roast", "Loc Country", "Origin", "100g_USD", "Rating", "Review Date", "Review"),
            show="headings"
        )

        v_scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=self.results_tree.yview)
        v_scrollbar.pack(side="right", fill="y")

        h_scrollbar = ttk.Scrollbar(tree_container, orient="horizontal", command=self.results_tree.xview)
        h_scrollbar.pack(side="bottom", fill="x")

        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.results_tree.pack(side="left", fill="both", expand=True)

        self.results_tree.heading("Name", text="Name", command=lambda: self.sort_treeview("Name", False))
        self.results_tree.heading("Roaster", text="Roaster", command=lambda: self.sort_treeview("Roaster", False))
        self.results_tree.heading("Roast", text="Roast", command=lambda: self.sort_treeview("Roast", False))
        self.results_tree.heading("Loc Country", text="Loc Country", command=lambda: self.sort_treeview("Loc Country", False))
        self.results_tree.heading("Origin", text="Origin", command=lambda: self.sort_treeview("Origin", False))
        self.results_tree.heading("100g_USD", text="100g_USD", command=lambda: self.sort_treeview("100g_USD", False))
        self.results_tree.heading("Rating", text="Rating", command=lambda: self.sort_treeview("Rating", False))
        self.results_tree.heading("Review Date", text="Review Date", command=lambda: self.sort_treeview("Review Date", False))
        self.results_tree.heading("Review", text="Review", command=lambda: self.sort_treeview("Review", False))

        self.results_tree.column("Name", width=150)
        self.results_tree.column("Roaster", width=100)
        self.results_tree.column("Roast", width=100)
        self.results_tree.column("Loc Country", width=100)
        self.results_tree.column("Origin", width=100)
        self.results_tree.column("100g_USD", width=80)
        self.results_tree.column("Rating", width=80)
        self.results_tree.column("Review Date", width=100)
        self.results_tree.column("Review", width=600)

    def update_conditions(self, selected_col):
        if selected_col == "review":
            if self.review_var.get():
                self.review_label.grid()
                self.review_entry.grid()
                self.num_neighbors_label.grid()
                self.num_neighbors_entry.grid()
            else:
                self.review_label.grid_remove()
                self.review_entry.grid_remove()
                self.num_neighbors_label.grid_remove()
                self.num_neighbors_entry.grid_remove()
            return

        is_selected = self.attr_vars[selected_col].get()
        if is_selected:
            if len(self.selected_attributes) >= 4:
                messagebox.showinfo("Limit Reached",
                                    "You can select up to 4 attributes. Please deselect another attribute first.")
                self.attr_vars[selected_col].set(False)
                return

            self.selected_attributes.append(selected_col)
        else:
            self.selected_attributes.remove(selected_col)

        for col in self.available_columns:
            if col in self.selected_attributes:
                self.cond_labels[col].grid()
                if col in self.numeric_attributes:
                    self.min_entries[col].grid()
                    self.max_entries[col].grid()
                else:
                    self.single_entries[col].grid()
            else:
                self.cond_labels[col].grid_remove()
                if col in self.numeric_attributes:
                    self.min_entries[col].grid_remove()
                    self.max_entries[col].grid_remove()
                else:
                    self.single_entries[col].grid_remove()

    def perform_search(self):
        self.selected_attributes = [col for col, var in self.attr_vars.items() if var.get()]
        if len(self.selected_attributes) > 4:
            messagebox.showerror("Error", "You can select at most 4 attributes.")
            return

        self.conditions = {}
        for col in self.selected_attributes:
            if col in self.numeric_attributes:
                min_value = self.min_entries[col].get().strip()
                max_value = self.max_entries[col].get().strip()

                if min_value or max_value:
                    self.conditions[col] = (
                        float(min_value) if min_value else None,
                        float(max_value) if max_value else None
                    )
            else:
                value = self.single_entries[col].get().strip()
                if value:
                    self.conditions[col] = value

        review_keywords = None
        num_neighbors = None
        if self.review_var.get():
            review_keywords = self.review_entry.get().strip()
            num_neighbors = self.num_neighbors_entry.get().strip()

            if not review_keywords:
                messagebox.showerror("Error", "Please enter review keywords.")
                return

            if not num_neighbors:
                messagebox.showerror("Error", "Please enter the number of nearest neighbors.")
                return

            num_neighbors = int(num_neighbors)

        if not self.conditions and not review_keywords:
            messagebox.showerror("Error", "No valid conditions provided.")
            return

        results = kdtree_main(self.selected_attributes, self.conditions, review_keywords, num_neighbors)
        print("Results from kdtree_main:", results)

        self.display_results(results)

    def display_results(self, results):
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)

        if results:
            for row in results:
                self.results_tree.insert("", "end", values=row)
        else:
            messagebox.showinfo("No Results", "No results match the given conditions.")

    def sort_treeview(self, col, reverse):
        rows = [(self.results_tree.set(item, col), item) for item in self.results_tree.get_children("")]
        try:
            rows.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            rows.sort(key=lambda t: t[0], reverse=reverse)

        for index, (_, item) in enumerate(rows):
            self.results_tree.move(item, "", index)

        self.results_tree.heading(col, command=lambda: self.sort_treeview(col, not reverse))

import tkinter as tk
from tkinter import ttk, messagebox
from rtree import rtree_main  # Import your existing R-tree main function


class RTreeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("R-Tree Search GUI")

        # Open the window in front of other apps
        self.root.lift()  # Bring the window to the top
        self.root.attributes('-topmost', True)  # Make the window stay on top
        self.root.after_idle(self.root.attributes, '-topmost', False)  # Allow it to be lowered after opening

        # Open the window in full window mode (maximized)
        self.root.state('zoomed')

        # Available attributes
        self.available_columns = ["100g_USD", "rating", "review_date", "roaster", "roast", "loc_country", "origin"]

        # Selected attributes and conditions
        self.selected_attributes = []
        self.conditions = {}

        # GUI Components
        self.create_widgets()

    def create_widgets(self):
        # Frame for attribute selection
        attr_frame = ttk.LabelFrame(self.root, text="Select attributes (up to 4)")
        attr_frame.pack(fill="x", padx=10, pady=10)

        # Checkboxes for attribute selection
        self.attr_vars = {}
        for i, col in enumerate(self.available_columns):
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(attr_frame, text=col, variable=var, command=lambda c=col: self.update_conditions(c))
            chk.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=2)
            self.attr_vars[col] = var

        # Frame for condition input
        self.cond_frame = ttk.LabelFrame(self.root, text="Enter conditions. You can have multiple conditions separated "
                                                         "by commas. Conditions for the numeric attributes have to be "
                                                         "in the form of <=, > etc. Dates are in YYYYMM format as a "
                                                         "number")
        self.cond_frame.pack(fill="x", padx=10, pady=10)

        # Entry fields for conditions (initially hidden)
        self.cond_labels = {}  # Dictionary to store labels
        self.cond_entries = {}  # Dictionary to store entry fields
        for i, col in enumerate(self.available_columns):
            label = ttk.Label(self.cond_frame, text=f"{col}:")
            label.grid(row=i, column=0, padx=5, pady=2, sticky="e")
            entry = ttk.Entry(self.cond_frame)
            entry.grid(row=i, column=1, padx=5, pady=2, sticky="w")
            self.cond_labels[col] = label
            self.cond_entries[col] = entry
            label.grid_remove()  # Hide the label initially
            entry.grid_remove()  # Hide the entry initially

        # Search button
        search_btn = ttk.Button(self.root, text="Search", command=self.perform_search)
        search_btn.pack(pady=10)

        # Frame for displaying results
        self.results_frame = ttk.LabelFrame(self.root, text="Search results. You can tap on the headings to sort the "
                                                            "table based on the attribute you want")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Create a container frame for the Treeview and Scrollbar
        tree_container = ttk.Frame(self.results_frame)
        tree_container.pack(fill="both", expand=True)

        # Treeview widget for displaying results
        self.results_tree = ttk.Treeview(
            tree_container,
            columns=("Name", "Roaster", "Roast", "Loc Country", "Origin", "100g_USD", "Rating", "Review Date"),
            show="headings"
        )
        self.results_tree.pack(side="left", fill="both", expand=True)

        # Add a vertical scrollbar
        scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=self.results_tree.yview)
        scrollbar.pack(side="right", fill="y")

        # Configure the Treeview to use the scrollbar
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Set column headings
        self.results_tree.heading("Name", text="Name", command=lambda: self.sort_treeview("Name", False))
        self.results_tree.heading("Roaster", text="Roaster", command=lambda: self.sort_treeview("Roaster", False))
        self.results_tree.heading("Roast", text="Roast", command=lambda: self.sort_treeview("Roast", False))
        self.results_tree.heading("Loc Country", text="Loc Country", command=lambda: self.sort_treeview("Loc Country", False))
        self.results_tree.heading("Origin", text="Origin", command=lambda: self.sort_treeview("Origin", False))
        self.results_tree.heading("100g_USD", text="100g_USD", command=lambda: self.sort_treeview("100g_USD", False))
        self.results_tree.heading("Rating", text="Rating", command=lambda: self.sort_treeview("Rating", False))
        self.results_tree.heading("Review Date", text="Review Date", command=lambda: self.sort_treeview("Review Date", False))

        # Set column widths
        self.results_tree.column("Name", width=150)
        self.results_tree.column("Roaster", width=100)
        self.results_tree.column("Roast", width=100)
        self.results_tree.column("Loc Country", width=100)
        self.results_tree.column("Origin", width=100)
        self.results_tree.column("100g_USD", width=80)
        self.results_tree.column("Rating", width=80)
        self.results_tree.column("Review Date", width=100)

    def update_conditions(self, selected_col):
        """Update the visibility of condition entry fields based on selected attributes."""
        # Get the current state of the checkbox
        is_selected = self.attr_vars[selected_col].get()

        # If the user is selecting an attribute
        if is_selected:
            # Check if the user has already selected 4 attributes
            if len(self.selected_attributes) >= 4:
                # Show a message box informing the user of the limit
                messagebox.showinfo("Limit Reached", "You can select up to 4 attributes. "
                                                     "Please deselect another attribute first.")
                # Uncheck the checkbox for the selected attribute
                self.attr_vars[selected_col].set(False)
                return

            # Add the selected attribute to the list
            self.selected_attributes.append(selected_col)
        else:
            # Remove the deselected attribute from the list
            self.selected_attributes.remove(selected_col)

        # Show condition entry fields for selected attributes and hide others
        for col in self.available_columns:
            if col in self.selected_attributes:
                self.cond_labels[col].grid()  # Show the label
                self.cond_entries[col].grid()  # Show the entry
            else:
                self.cond_labels[col].grid_remove()  # Hide the label
                self.cond_entries[col].grid_remove()  # Hide the entry

    def perform_search(self):
        # Get selected attributes
        self.selected_attributes = [col for col, var in self.attr_vars.items() if var.get()]
        if len(self.selected_attributes) > 4:
            messagebox.showerror("Error", "You can select at most 4 attributes.")
            return

        # Get conditions
        self.conditions = {}
        for col in self.selected_attributes:
            condition_input = self.cond_entries[col].get().strip()
            if condition_input:
                # Split conditions by comma
                condition_list = [cond.strip() for cond in condition_input.split(",")]
                self.conditions[col] = condition_list

        if not self.conditions:
            messagebox.showerror("Error", "No valid conditions provided.")
            return

        # Call the R-tree main function with selected attributes and conditions
        results = rtree_main(self.selected_attributes, self.conditions)
        print("Results from rtree_main:", results)  # Debug print

        # Display the results in the Treeview
        self.display_results(results)

    def display_results(self, results):
        """Display the search results in the Treeview."""
        # Clear existing results
        for row in self.results_tree.get_children():
            self.results_tree.delete(row)

        # Add new results
        if results:
            for row in results:
                self.results_tree.insert("", "end", values=row)
        else:
            messagebox.showinfo("No Results", "No results match the given conditions.")

    def sort_treeview(self, col, reverse):
        """Sort the Treeview by the selected column."""
        # Get all rows from the Treeview
        rows = [(self.results_tree.set(item, col), item) for item in self.results_tree.get_children("")]

        # Sort rows based on the column values
        try:
            # Try to sort as numbers
            rows.sort(key=lambda t: float(t[0]), reverse=reverse)
        except ValueError:
            # If sorting as numbers fails, sort as strings
            rows.sort(key=lambda t: t[0], reverse=reverse)

        # Rearrange items in the Treeview
        for index, (_, item) in enumerate(rows):
            self.results_tree.move(item, "", index)

        # Reverse the sort order for the next click
        self.results_tree.heading(col, command=lambda: self.sort_treeview(col, not reverse))

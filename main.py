import tkinter as tk
from rtree_gui import RTreeGUI
from kdtree_gui import KDTreeGUI
from quadtree_gui import QuadtreeGUI
from rangetree_gui import RangeTreeGUI


class MainGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tree Selection GUI")

        # Set the background color of the root window to very dark grey
        self.root.configure(bg="#1E1E1E")

        # Open the window in full window mode (maximized)
        self.root.state('zoomed')

        # Create the main header
        main_header = tk.Label(
            root, text="MULTI - DIMENSIONAL DATA STRUCTURES PROJECT",
            font=("Arial", 28), bg="#1E1E1E", fg="white"  # Large white text
        )
        main_header.pack(pady=(50, 10))  # Add padding above and below

        # Create the subheader
        subheader = tk.Label(
            root, text="Select the tree structure you want to use for filtering the coffee reviews data",
            font=("Arial", 18, "italic"), bg="#1E1E1E", fg="white"  # Smaller white text
        )
        subheader.pack(pady=(0, 50))  # Add padding below

        # Create buttons for each tree type
        button_frame = tk.Frame(root, bg="#1E1E1E")  # Set background color of the frame
        button_frame.pack(pady=20)

        # Button for R-Tree
        rtree_button = tk.Button(
            button_frame, text="R-Tree", font=("Arial", 18), width=20,
            bg="#90EE90", fg="black", cursor="hand2",  # Light green, black text, hand cursor
            command=self.open_rtree_gui
        )
        rtree_button.pack(pady=10)
        self.add_hover_effect(rtree_button, "#90EE90", "#32CD32")  # Brighter green on hover

        # Button for KD-Tree
        kdtree_button = tk.Button(
            button_frame, text="KD-Tree", font=("Arial", 18), width=20,
            bg="#FFFF99", fg="black", cursor="hand2",  # Bright yellow, black text, hand cursor
            command=self.open_kdtree_gui
        )
        kdtree_button.pack(pady=10)
        self.add_hover_effect(kdtree_button, "#FFFF99", "#FFD700")  # Brighter gold on hover

        # Button for QuadTree
        quadtree_button = tk.Button(
            button_frame, text="QuadTree", font=("Arial", 18), width=20,
            bg="#FF9999", fg="black", cursor="hand2",  # Bright red, black text, hand cursor
            command=self.open_quadtree_gui
        )
        quadtree_button.pack(pady=10)
        self.add_hover_effect(quadtree_button, "#FF9999", "#FF0000")  # More vibrant red on hover

        # Button for Range Tree
        rangetree_button = tk.Button(
            button_frame, text="Range Tree", font=("Arial", 18), width=20,
            bg="#99CCFF", fg="black", cursor="hand2",  # Bright blue, black text, hand cursor
            command=self.open_rangetree_gui
        )
        rangetree_button.pack(pady=10)
        self.add_hover_effect(rangetree_button, "#99CCFF", "#1E90FF")  # Brighter dodger blue on hover

        # Add an exit button to close the window
        exit_button = tk.Button(
            root, text="Exit", font=("Arial", 14), bg="#FF0000", fg="white",
            cursor="hand2", command=self.root.destroy  # Hand cursor for exit button
        )
        exit_button.pack(pady=20)
        self.add_hover_effect(exit_button, "#FF0000", "#8B0000")  # Darker red on hover

        # Add the "Developed by" section in the right-down corner
        developed_by_frame = tk.Frame(root, bg="#1E1E1E")  # Frame to hold the labels
        developed_by_frame.place(relx=1.0, rely=1.0, anchor="se", x=-20, y=-20)  # Place in bottom-right corner

        # Add "DEVELOPED BY" label (centered in the frame)
        developed_by_header = tk.Label(
            developed_by_frame, text="DEVELOPED BY",
            font=("Comic Sans MS", 12, "bold"), bg="#1E1E1E", fg="white"
        )
        developed_by_header.pack(pady=(0, 5))  # Add padding below the header

        # Add the list of developers
        developed_by_text = (
            "1093356 - Zannis Georgios\n"
            "1093359 - Zafeiris Vasileios\n"
            "1093516 - Psaltiras Panagiotis\n"
            "1097454 - Vasilopoulos Panagiotis"
        )
        developed_by_label = tk.Label(
            developed_by_frame, text=developed_by_text,
            font=("Comic Sans MS", 12, "italic"), bg="#1E1E1E", fg="white", justify="left"
        )
        developed_by_label.pack()  # No additional padding needed

    def add_hover_effect(self, button, default_color, hover_color):
        """Add hover effects to a button."""
        # Change color when mouse enters the button
        button.bind("<Enter>", lambda event: button.config(bg=hover_color))

        # Revert color when mouse leaves the button
        button.bind("<Leave>", lambda event: button.config(bg=default_color))

    def open_rtree_gui(self):
        """Open the R-Tree GUI."""
        rtree_root = tk.Toplevel(self.root)
        app = RTreeGUI(rtree_root)

    def open_kdtree_gui(self):
        """Open the KD-Tree GUI."""
        kdtree_root = tk.Toplevel(self.root)
        app = KDTreeGUI(kdtree_root)

    def open_quadtree_gui(self):
        """Open the QuadTree GUI."""
        quadtree_root = tk.Toplevel(self.root)
        app = QuadtreeGUI(quadtree_root)

    def open_rangetree_gui(self):
        """Open the Range Tree GUI."""
        rangetree_root = tk.Toplevel(self.root)
        app = RangeTreeGUI(rangetree_root)


def main():
    root = tk.Tk()
    app = MainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

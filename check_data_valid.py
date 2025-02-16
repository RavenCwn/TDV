from glob import glob


# AssertionError: Not enough front_rgb images in all_data/place_cups/variation0/episodes/episode1
# AssertionError: Not enough front_rgb images in all_data/turn_tap/variation0/episodes/episode79

# AssertionError: Expected 125 episodes in all_data/place_cups, but got 124

if __name__ == '__main__':
    # Get all the files in the data directory
    task_dir = glob("all_data/*")
    
    for task in sorted(task_dir):
        # Get all the files in the task directory
        task_files = glob(task + "/variation0/episodes/*")

        assert len(task_files) == 125, f"Expected 125 episodes in {task}, but got {len(task_files)}"

        for episode in sorted(task_files):
            front_rgb = glob(episode + "/front_rgb/*")
            assert len(front_rgb) > 30, f"Not enough front_rgb images in {episode}"
            
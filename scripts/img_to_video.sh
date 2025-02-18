ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/test_save/close_jar/0000/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p save0.mp4

ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/test_save/close_jar/0001/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p save1.mp4

ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/test_save/close_jar/0002/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p save2.mp4

ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/test_save/close_jar/0003/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p save3.mp4

ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/test_save/close_jar/0004/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p save4.mp4

ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/two_save/close_jar/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p save4.mp4

ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/two_save_1/insert_onto_square_peg/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p save5.mp4

/home/c4090/trajectory_diff/coordiff/combine_var_data/val/turn_tap/episode_0021/images/front_rgb

ffmpeg -framerate 30 -pattern_type glob -i "/home/c4090/trajectory_diff/coordiff/combine_var_data/val/turn_tap/episode_0021/images/front_rgb/*.png" -c:v mpeg4 -pix_fmt yuv420p turn_tap.mp4

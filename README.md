<h1 style='text-align: center;'> <b> Eye to Screen Distance Measurement With Computer Vision 👀💻 </b></h1>

- This project is to calculate the distance between your eyes and laptop screen using the webcam.

- The concept is that an object will appear smaller once it is further away from the camera and vice versa.

- The distance between your eyes (pixel) will be used as the reference object and will get converted into the distance (cm) to the laptop screen.

- **[Ergonomics tip:](https://www.ergotron.com/zh-sg/ergonomics/ergonomic-equation)** laptop screen should be positioned at least **20 inches (51 cm)** away from your eyes.

![GIF](demo\video-1.gif)

---

<h2><b> How to run this app </b><img src="https://emojis.slackmojis.com/emojis/images/1600706728/10521/meow_code.gif?1600706728" width="25"/> </h2>

#### **📝Note: For quick testing jump straight to Step-2**

### **Step-1:**

- In **distance_calculator.py**, create an instance of **DistanceCalculator** class and call **run_config()** instance method

```python
if __name__ == '__main__':
    eye_screen_distance = DistanceCalculator()
    eye_screen_distance.run_config()
```

- The webcam will start working and the distance between the user's eyes (pixel) will show up.

- Using a measuring tape, take a few measurements (cm) between your eye and laptop screen, then jot them down along with the corresponding distance between eyes (pixel).

- Take a minimum of five measurements and different locations and save them to **distance_xy.csv**.

- click **letter K** on the keyboard to stop the webcam.

- This is a one-time step and no need to do it again after updating **distance_xy.csv**.

<img src="demo\Pic1.png"/>
<img src="demo\Pic2.png"/>

### **Step-2:**

- In distance_calculator.py, read **distance_xy.csv** file, create an instance of **DistanceCalculator** class and call **calculate_distance()** instance method

```python
if __name__ == '__main__':
    distance_df = pd.read_csv('distance_xy.csv')
    eye_screen_distance = DistanceCalculator()
    eye_screen_distance.calculate_distance(distance_df['distance_pixel'], distance_df['distance_cm'])
```

- click **letter K** on the keyboard to stop the webcam.

---

<h2><b> Author </b><img src="https://emojis.slackmojis.com/emojis/images/1531849430/4246/blob-sunglasses.gif?1531849430" width="25"/> </h2>

- Name: Ahmed Abdulrahman
- Researcher/Data Scientist/Engineer
- Email: ahmedabdulrahman419@gmail.com

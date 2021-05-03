
class Colors:

    def __init__(self):
        self.dark_mode = False

        self.page_bg_light = '#F0FFFF'
        self.plot_bg_light = '#ffffff'

        self.page_bg_dark = 'rgb(50, 50, 50)'
        self.plot_bg_dark = '#A9A9A9'

    def get_page_bg(self):
        if self.dark_mode:
            return self.page_bg_dark
        else:
            return self.page_bg_light

    def get_plot_bg(self):
        if self.dark_mode:
            return self.plot_bg_dark
        else:
            return self.plot_bg_light

    def set_dark_mode(self, dark_mode):
        self.dark_mode = dark_mode

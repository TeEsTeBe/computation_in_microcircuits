import math
from matplotlib import pyplot as pl, animation


class ActivityAnimator(object):
    """
    ActivityIllustrator(spike_list, vm_list=None, populations=None, ids=None)
    Class for animating various spiking activities of SpikeList objects or populations.

    There is a single public function, animate_activity(...), that controls all the animations. For performance reasons,
    make sure to animate the activities of only a reduced number of neurons (~100 - 300~).
    Possible animations:
        raster
        mean rate
        membrane voltage
        trajectory

    Input:
        spike_list
        ids: None or a list of lists, grouped by ids (for coloring, etc.)

    Examples:
        >> ai = viz.ActivityIllustrator(spikelist)

        >> ai.animate_activity(time_interval=100, time_window=100, fps=60, frame_res=0.25, save=True,
                                filename="/path/to/saved/file", activities=["raster", "rate"])

    """
    def __init__(self, spike_list, vm_list=None, populations=None, ids=None):
        self.spike_list 	= spike_list
        self.vm_list 		= vm_list
        self.populations 	= populations
        self.ids 			= ids

    def __anim_frame_raster(self, ax=None, start=None, stop=None, colors=['b'], shift_only=False, ax_props={}):
        """
        Display the contents of the spike list for the chosen neurons as a raster plot.

        :param ax:
        :param start:
        :param stop:
        :param colors:
        :param shift_only:
        :param ax_props:
        :return:
        """
        ax.clear()

        if not shift_only:
            self.raster_fr_data = []
            tt = self.sliced_spike_list

            if self.ids is None:
                times 	= tt.raw_data()[:, 0]
                neurons = tt.raw_data()[:, 1]
                ax.plot(times, neurons, '.', color=colors[0])
                self.raster_fr_data.append((times, neurons, '.', colors[0]))
            else:
                assert isinstance(self.ids, list), "Gids should be a list"
                for n, group_ids in enumerate(self.ids):
                    group = tt.id_slice(group_ids)
                    times = group.raw_data()[:, 0]
                    neurons = group.raw_data()[:, 1]
                    ax.plot(times, neurons, '.', color=colors[n])
                    self.raster_fr_data.append((times, neurons, '.', colors[n]))
        else:
            assert self.raster_fr_data is not None and not isinstance(self.raster_fr_data, str), \
                "Previous raster frame data empty, something's gone south..."
            for fd in self.raster_fr_data:
                times, neurons, format_, color = fd
                ax.plot(times, neurons, '.', color=color)

        ax.set_xlim([start, stop])
        ax.set(**ax_props)

    def __anim_frame_mean_rate(self, ax=None, start=None, stop=None, colors=['b'], shift_only=False, dt=1., ax_props={}):
        """

        :param ax:
        :param start:
        :param stop:
        :param colors:
        :param shift_only:
        :param dt:
        :param ax_props:
        :return:
        """
        ax.clear()
        # only precompute it max_rate once, for size of rate subplot
        if self.max_rate is None:
            firing_rates = self.spike_list.firing_rate(dt, average=True)
            self.max_rate = max(firing_rates)
            self.min_rate = min(firing_rates)

        if not shift_only:
            self.rate_fr_data = []
            tt = self.sliced_spike_list

            if self.ids is None:
                time = tt.time_axis(dt)[:-1]
                rate = tt.firing_rate(dt, average=True)
                ax.plot(time, rate, color=colors[0])
                self.rate_fr_data.append((time, rate, colors[0]))
            else:
                assert isinstance(self.ids, list), "Gids should be a list"
                for n, group_ids in enumerate(self.ids):
                    group = tt.id_slice(group_ids)
                    time = group.time_axis(dt)[:-1]
                    rate = group.firing_rate(dt, average=True)
                    ax.plot(time, rate, colors[n])
                    self.rate_fr_data.append((time, rate, colors[n]))
        else:
            assert self.rate_fr_data is not None and not isinstance(self.rate_fr_data, str), \
                "Previous rate frame data empty, something's gone south..."
            for fd in self.rate_fr_data:
                time, rate, color = fd
                ax.plot(time, rate, color=color)

        ax.set(ylim=[self.min_rate-1, self.max_rate+1], xlim=[start, stop])
        ax.set(**ax_props)

    def __plot_vm(self, ax=None, time_interval=None, with_spikes=True):
        pass

    def __plot_trace(self, ax=None, time_interval=None, dt=1.):
        pass

    def __plot_trajectory(self, ax=None, start=None, stop=None, colors=['b'], shift_only=False, dt=1., ax_props={}):
        pass

    def __has_activity(self, activities, key):
        return bool(activities is not None and key in activities)

    def __step_binned_time(self, start=0., time_step=0.1, time_window=100):
        while True:
            yield (start, start + time_window - 1)
            start += time_step

    def animate_activity(self, time_interval=None, time_window=None, frame_res=0.25, fps=60, sim_res=1.0,
                         filename="animation", activities=None, colors=['b'], save=False, display=False):
        """
        Main function that exposes all the animations. It uses animation.FuncAnimation to plot each frame separately,
        which are then saved in .mp4 format. Every call to animate_activity creates a single figure containing all the
        activities that were passed as parameters. Use the frame_res and fps parameters to create smooth animations.


        :param time_interval: time interval in msec between each frame of the animation [USEFUL ONLY WHEN DISPLAYING]
        :param time_window: size of time window for each frame
        :param frame_res: resolution of the time axis for the animation, i.e., how many if simulation resolution=1.0 and
            dt=0.25 there will be 4 frames for each second on the time axis.
        :param save:
        :param fps: frames per second for the video encoding
        :param sim_res: resolution of the simulation (for spike timing)
        :param filename:
        :param activities: list of strings; possible values: ["raster", "rate"]
        :param colors: list of matplotlib colors; numbers of colors must match the number of populations (id groups)
        :return:
        """
        print("Started animating spiking activities...")

        def animate(i):
            # TODO check if performance can be increased by using blit=True.. redesign needed in that case
            start_t, stop_t = next(mwt) # start and stop time of spike window
            self.sliced_spike_list = self.spike_list.time_slice(start_t, stop_t)

            # only draw new plot at every simulation resolution (when new spikes can appear), otherwise just shift axes
            shift_only = bool(self.cnt % int(sim_res / frame_res) != 0)

            if self.__has_activity(activities, "raster"):
                self.__anim_frame_raster(ax_raster, start_t, stop_t, colors, shift_only,
                                         {"xlabel": "Time [ms]", "ylabel": "Neurons"})

            if self.__has_activity(activities, "rate"):
                self.__anim_frame_mean_rate(ax_rate, start_t, stop_t, colors, shift_only,
                                            ax_props={"xlabel": "Time", "ylabel": "Rate [Hz]"})

            self.cnt += 1

        # some initial checks
        if self.ids is not None and len(self.ids) > 1:
            assert len(self.ids) == len(colors), "Number of population groups and #colors must match!"

        time_axis 	= self.spike_list.time_axis(time_bin=frame_res)
        nr_frames 	= int(math.floor((len(time_axis) * frame_res) - time_window + 1) / frame_res)
        mwt 		= self.__step_binned_time(min(time_axis), frame_res, time_window)
        self.cnt 	= 0  # counter for #calls to animation function (animate)

        fig = pl.figure()
        if self.__has_activity(activities, "raster"):
            ax_raster 	= pl.subplot2grid((30, 1), (0, 0), rowspan=23, colspan=1)
        if self.__has_activity(activities, "rate"):
            ax_rate 	= pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1)
        if self.__has_activity(activities, "trajectory"):
            ax_trajectory = pl.subplot(111, projection='3d')#pl.subplot2grid((30, 1), (24, 0), rowspan=5, colspan=1)
            ax_trajectory.grid(False)

        self.raster_fr_data = None
        self.rate_fr_data 	= None
        self.trajectory_fr_data = None
        self.max_rate 		= None
        self.sliced_spike_list = None

        # run animation
        ani = animation.FuncAnimation(fig, animate, interval=time_interval, frames=nr_frames - 1, repeat=False)
        if save:
            ani.save(filename + '.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
        if display:
            pl.show()
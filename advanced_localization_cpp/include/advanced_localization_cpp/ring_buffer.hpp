#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace advanced_localization
{

template<typename T>
class TimeIndexedRingBuffer
{
public:
  explicit TimeIndexedRingBuffer(std::size_t capacity)
  : capacity_(capacity)
  {
    if (capacity_ < 2) {
      throw std::invalid_argument("capacity must be at least 2");
    }
  }

  void append(double timestamp_s, const T & item)
  {
    auto it = std::upper_bound(times_.begin(), times_.end(), timestamp_s);
    auto index = static_cast<std::size_t>(std::distance(times_.begin(), it));
    times_.insert(it, timestamp_s);
    items_.insert(items_.begin() + static_cast<std::ptrdiff_t>(index), item);

    const auto overflow = static_cast<long>(times_.size()) - static_cast<long>(capacity_);
    if (overflow > 0) {
      times_.erase(times_.begin(), times_.begin() + overflow);
      items_.erase(items_.begin(), items_.begin() + overflow);
    }
  }

  bool latest(double & timestamp_s, T & item) const
  {
    if (times_.empty()) {
      return false;
    }
    timestamp_s = times_.back();
    item = items_.back();
    return true;
  }

  bool at_or_before(double timestamp_s, double & out_timestamp_s, T & item) const
  {
    if (times_.empty()) {
      return false;
    }
    auto it = std::upper_bound(times_.begin(), times_.end(), timestamp_s);
    if (it == times_.begin()) {
      return false;
    }
    --it;
    const auto index = static_cast<std::size_t>(std::distance(times_.begin(), it));
    out_timestamp_s = times_[index];
    item = items_[index];
    return true;
  }

  bool at_nearest(double timestamp_s, double & out_timestamp_s, T & item) const
  {
    if (times_.empty()) {
      return false;
    }
    auto right = std::lower_bound(times_.begin(), times_.end(), timestamp_s);
    std::size_t index = 0;
    if (right == times_.begin()) {
      index = 0;
    } else if (right == times_.end()) {
      index = times_.size() - 1;
    } else {
      const auto right_index = static_cast<std::size_t>(std::distance(times_.begin(), right));
      const auto left_index = right_index - 1;
      const double left_delta = std::abs(times_[left_index] - timestamp_s);
      const double right_delta = std::abs(times_[right_index] - timestamp_s);
      index = left_delta <= right_delta ? left_index : right_index;
    }
    out_timestamp_s = times_[index];
    item = items_[index];
    return true;
  }

  std::vector<std::pair<double, T>> after(double timestamp_s) const
  {
    std::vector<std::pair<double, T>> result;
    auto it = std::upper_bound(times_.begin(), times_.end(), timestamp_s);
    for (; it != times_.end(); ++it) {
      const auto index = static_cast<std::size_t>(std::distance(times_.begin(), it));
      result.emplace_back(times_[index], items_[index]);
    }
    return result;
  }

  void clear_after(double timestamp_s)
  {
    auto keep_end = std::upper_bound(times_.begin(), times_.end(), timestamp_s);
    const auto index = static_cast<std::size_t>(std::distance(times_.begin(), keep_end));
    times_.erase(times_.begin() + static_cast<std::ptrdiff_t>(index), times_.end());
    items_.erase(items_.begin() + static_cast<std::ptrdiff_t>(index), items_.end());
  }

  void clear()
  {
    times_.clear();
    items_.clear();
  }

  std::size_t size() const { return times_.size(); }

private:
  std::size_t capacity_;
  std::vector<double> times_;
  std::vector<T> items_;
};

}  // namespace advanced_localization
